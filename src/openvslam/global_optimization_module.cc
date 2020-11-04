#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/match/fuse.h"
#include "openvslam/util/converter.h"

#include <spdlog/spdlog.h>

namespace openvslam {

global_optimization_module::global_optimization_module(data::map_database* map_db, data::bow_database* bow_db,
                                                       data::bow_vocabulary* bow_vocab, const bool fix_scale)
    : loop_detector_(new module::loop_detector(bow_db, bow_vocab, fix_scale)),
      loop_bundle_adjuster_(new module::loop_bundle_adjuster(map_db)),
      graph_optimizer_(new optimize::graph_optimizer(map_db, fix_scale)) {
    spdlog::debug("CONSTRUCT: global_optimization_module");
}

global_optimization_module::~global_optimization_module() {
    abort_loop_BA();
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
    }
    spdlog::debug("DESTRUCT: global_optimization_module");
}

void global_optimization_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void global_optimization_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    loop_bundle_adjuster_->set_mapping_module(mapper);
}

void global_optimization_module::enable_loop_detector() {
    spdlog::info("enable loop detector");
    loop_detector_->enable_loop_detector();
}

void global_optimization_module::disable_loop_detector() {
    spdlog::info("disable loop detector");
    loop_detector_->disable_loop_detector();
}

bool global_optimization_module::loop_detector_is_enabled() const {
    return loop_detector_->is_enabled();
}

//回环检测和全局优化线程，用来进行回环检测，当回环检测成功之后调用全局优化算法进行全局优化
void global_optimization_module::run() {
    spdlog::info("start global optimization module");

    is_terminated_ = false;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));//当前线程休眠5毫秒

        // check if termination is requested
        //检测是否通过pangolin界面上的按钮发起了结束slam的请求
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if pause is requested
        //检测是否通过pangolin界面上的按钮发起了暂停slam的请求
        if (pause_is_requested()) {
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        //检测是否通过pangolin界面上的reset按钮发起了重置slam的请求
        if (reset_is_requested()) {
            // reset and continue
            reset();
            continue;
        }

        // if the queue is empty, the following process is not needed
        //如果当前回环检测备选keyframe的链表中没有keyframe，那么就不用执行后面的回环检测了
        if (!keyframe_is_queued()) {
            continue;
        }

        // dequeue the keyframe from the queue -> cur_keyfrm_
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            cur_keyfrm_ = keyfrms_queue_.front();//取出链表中的第一个keyframe
            keyfrms_queue_.pop_front();//删除链表第一个元素
        }

        // not to be removed during loop detection and correction
        cur_keyfrm_->set_not_to_be_erased();//回环检测期间，不得对当前keyframe执行删除

        // pass the current keyframe to the loop detector
        loop_detector_->set_current_keyframe(cur_keyfrm_);//设置回环检测器的当前keyframe

        // detect some loop candidate with BoW
        if (!loop_detector_->detect_loop_candidates()) {//通过cur_keyfrm_到数据库中去寻找回环候选帧，如入没有满足条件的keyframe，则结束此次循环
            // could not find
            // allow the removal of the current keyframe
            cur_keyfrm_->set_to_be_erased();//如果没有找到满足条件的回环候选帧，那么当前keyframe就可以进行删除操作(具体删除还是不删除需要看关键帧的判断条件)
            continue;
        }

        // validate candidates and select ONE candidate from them
        if (!loop_detector_->validate_candidates()) {//验证当前回环检测是否成功，如果不成功则结束此次循环
            // could not find
            // allow the removal of the current keyframe
            cur_keyfrm_->set_to_be_erased();//如果查找出来的回环帧不满足回环条件，当前帧就可以值执行删除操作
            continue;
        }

        correct_loop();//执行回环修正
    }

    spdlog::info("terminate global optimization module");
}

void global_optimization_module::queue_keyframe(data::keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    if (keyfrm->id_ != 0) {
        keyfrms_queue_.push_back(keyfrm);
    }
}

bool global_optimization_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return (!keyfrms_queue_.empty());
}

void global_optimization_module::correct_loop() {
    auto final_candidate_keyfrm = loop_detector_->get_selected_candidate_keyframe();//取出最终检测到的回环帧

    spdlog::info("detect loop: keyframe {} - keyframe {}", final_candidate_keyfrm->id_, cur_keyfrm_->id_);
    loop_bundle_adjuster_->count_loop_BA_execution();//记录回环检测以后执行BA优化的次数

    // 0. pre-processing

    // 0-1. stop the mapping module and the previous loop bundle adjuster

    // pause the mapping module
    mapper_->request_pause();//暂停建图模块的工作
    // abort the previous loop bundle adjuster
    //如果之前检测到了回环，并且正在执行全局优化，则放弃之前的优化
    if (thread_for_loop_BA_ || loop_bundle_adjuster_->is_running()) {
        abort_loop_BA();
    }
    // wait till the mapping module pauses
    while (!mapper_->is_paused()) {//等待mapper模块确实暂停了
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // 0-2. update the graph

    cur_keyfrm_->graph_node_->update_connections();//当前关键帧可能发生了一些数据变化，所以需要进行更新

    // 1. compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector
    //    then, the covisibilities are moved to the corrected positions
    //    finally, landmarks observed in them are also moved to the correct position using the camera poses before and after camera pose correction

    // acquire the covisibilities of the current keyframe
    std::vector<data::keyframe*> curr_neighbors = cur_keyfrm_->graph_node_->get_covisibilities();//获得当前关键帧的共视关键帧
    curr_neighbors.push_back(cur_keyfrm_);//把自己也加进去

    // Sim3 camera poses BEFORE loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_correction;//回环修正之前当前关键帧以及共视关键帧的Sim3变换
    // Sim3 camera poses AFTER loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_correction;//回环修正之后当前关键帧以及共视关键帧的Sim3变换

    const auto g2o_Sim3_cw_after_correction = loop_detector_->get_Sim3_world_to_current();//经过回环检测之后，当前关键帧相对于世界坐标系下的Sim3变换
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // camera pose of the current keyframe BEFORE loop correction
        const Mat44_t cam_pose_wc_before_correction = cur_keyfrm_->get_cam_pose_inv();//获取根据前端里程计计算出来的当前关键帧的变换矩阵，相对于世界坐标系

        // compute Sim3s BEFORE loop correction
        Sim3s_nw_before_correction = get_Sim3s_before_loop_correction(curr_neighbors);//构造当前关键帧的共视关键帧变换矩阵的Sim3形式
        // compute Sim3s AFTER loop correction
        //计算当前关键帧的共视关键帧回环修正之后的Sim3变换
        Sim3s_nw_after_correction = get_Sim3s_after_loop_correction(cam_pose_wc_before_correction, g2o_Sim3_cw_after_correction, curr_neighbors);

        // correct covibisibility landmark positions
        correct_covisibility_landmarks(Sim3s_nw_before_correction, Sim3s_nw_after_correction);
        // correct covisibility keyframe camera poses
        //将当前关键帧以及共视关键帧根据根据之前计算出来的变换关系，进行回环修正
        correct_covisibility_keyframes(Sim3s_nw_after_correction);
    }

    // 2. resolve duplications of landmarks caused by loop fusion

    const auto curr_match_lms_observed_in_cand = loop_detector_->current_matched_landmarks_observed_in_candidate();//当前帧和回环帧匹配上的地图点
    replace_duplicated_landmarks(curr_match_lms_observed_in_cand, Sim3s_nw_after_correction);

    // 3. extract the new connections created after loop fusion

    const auto new_connections = extract_new_connections(curr_neighbors);

    // 4. pose graph optimization

    graph_optimizer_->optimize(final_candidate_keyfrm, cur_keyfrm_, Sim3s_nw_before_correction, Sim3s_nw_after_correction, new_connections);

    // add a loop edge
    final_candidate_keyfrm->graph_node_->add_loop_edge(cur_keyfrm_);
    cur_keyfrm_->graph_node_->add_loop_edge(final_candidate_keyfrm);

    // 5. launch loop BA

    while (loop_bundle_adjuster_->is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
        thread_for_loop_BA_.reset(nullptr);
    }
    thread_for_loop_BA_ = std::unique_ptr<std::thread>(new std::thread(&module::loop_bundle_adjuster::optimize, loop_bundle_adjuster_.get(), cur_keyfrm_->id_));

    // 6. post-processing

    // resume the mapping module
    mapper_->resume();

    // set the loop fusion information to the loop detector
    loop_detector_->set_loop_correct_keyframe_id(cur_keyfrm_->id_);
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_before_loop_correction(const std::vector<data::keyframe*>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_loop_correction;

    for (const auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw = neighbor->get_cam_pose();
        // create Sim3 from SE3
        const Mat33_t& rot_nw = cam_pose_nw.block<3, 3>(0, 0);
        const Vec3_t& trans_nw = cam_pose_nw.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nw_before_correction(rot_nw, trans_nw, 1.0);
        Sim3s_nw_before_loop_correction[neighbor] = Sim3_nw_before_correction;
    }

    return Sim3s_nw_before_loop_correction;
}

/*!
 * 根据回环成功之后的Sim3变换，修正当前帧的所有共视关键帧的位姿
 * @param cam_pose_wc_before_correction 当前关键帧在修正之前的SE3变换，根据里程计获得的，相对于世界坐标系
 * @param g2o_Sim3_cw_after_correction 当前关键帧经过回环修正之后的Sim3变换，相对于世界坐标系
 * @param neighbors 当前关键帧的所有共视关键帧
 * @return 返回的是关键帧和其对应的修正之后的Sim3组成的map
 */
module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_after_loop_correction(const Mat44_t& cam_pose_wc_before_correction,
                                                                                          const g2o::Sim3& g2o_Sim3_cw_after_correction,
                                                                                          const std::vector<data::keyframe*>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_loop_correction;

    for (auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw_before_correction = neighbor->get_cam_pose();//当前相机变换矩阵，相对于相机坐标系
        // create the relative Sim3 from the current to `neighbor`
        const Mat44_t cam_pose_nc = cam_pose_nw_before_correction * cam_pose_wc_before_correction;//当前关键帧到共视关键帧的变换矩阵
        const Mat33_t& rot_nc = cam_pose_nc.block<3, 3>(0, 0);//旋转矩阵
        const Vec3_t& trans_nc = cam_pose_nc.block<3, 1>(0, 3);//平移矩阵
        const g2o::Sim3 Sim3_nc(rot_nc, trans_nc, 1.0);//构造Sim3变换
        // compute the camera poses AFTER loop correction of the neighbors
        const g2o::Sim3 Sim3_nw_after_correction = Sim3_nc * g2o_Sim3_cw_after_correction;//对当前共视关键帧进行回环修正
        Sim3s_nw_after_loop_correction[neighbor] = Sim3_nw_after_correction;
    }

    return Sim3s_nw_after_loop_correction;
}

/*!
 * 回环成功之后，对当前关键帧和共视关键帧的地图点进行矫正
 * @param Sim3s_nw_before_correction 当前关键帧修正之前的Sim3形式的变换矩阵
 * @param Sim3s_nw_after_correction 当前关键帧修正之后的Sim3变换矩阵
 */
void global_optimization_module::correct_covisibility_landmarks(const module::keyframe_Sim3_pairs_t& Sim3s_nw_before_correction,
                                                                const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {//对回环修正之后的当前关键帧以及其共视关键帧进行循环
        auto neighbor = t.first;//取出关键帧
        // neighbor->world AFTER loop correction
        const auto Sim3_wn_after_correction = t.second.inverse();//对Sim3变换求逆变换，获得相机到世界坐标系的Sim3变换
        // world->neighbor BEFORE loop correction
        const auto& Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);//回环修正之前的Sim3变换，世界到相机坐标系的变换

        const auto ngh_landmarks = neighbor->get_landmarks();//当前关键帧的地图点
        for (auto lm : ngh_landmarks) {
            if (!lm) {//空地图点？
                continue;
            }
            if (lm->will_be_erased()) {//坏点？
                continue;
            }

            // avoid duplication
            //如果当前地图点，之前已经进行过处理，则不再处理。主要原因是因为一个地图点可能被多次观测到
            if (lm->loop_fusion_identifier_ == cur_keyfrm_->id_) {
                continue;
            }
            lm->loop_fusion_identifier_ = cur_keyfrm_->id_;

            // correct position of `lm`
            const Vec3_t pos_w_before_correction = lm->get_pos_in_world();//地图点的世界坐标
            const Vec3_t pos_w_after_correction = Sim3_wn_after_correction.map(Sim3_nw_before_correction.map(pos_w_before_correction));//修正之后的地图点坐标
            lm->set_pos_in_world(pos_w_after_correction);//将当前landmark的坐标设置为修正之后的坐标
            // update geometry
            lm->update_normal_and_depth();//更新地图点的平均观测方向和尺度误差

            // record the reference keyframe used in loop fusion of landmarks
            lm->ref_keyfrm_id_in_loop_fusion_ = neighbor->id_;
        }
    }
}

void global_optimization_module::correct_covisibility_keyframes(const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const auto Sim3_nw_after_correction = t.second;

        const auto s_nw = Sim3_nw_after_correction.scale();
        const Mat33_t rot_nw = Sim3_nw_after_correction.rotation().toRotationMatrix();
        const Vec3_t trans_nw = Sim3_nw_after_correction.translation() / s_nw;
        const Mat44_t cam_pose_nw = util::converter::to_eigen_cam_pose(rot_nw, trans_nw);
        neighbor->set_cam_pose(cam_pose_nw);

        // update graph
        neighbor->graph_node_->update_connections();
    }
}

/*!
 * 当回环检测成功之后，当前帧以及共视关键帧对应的地图点与回环关键帧以及共视关键帧地图点有重复，需要检测这些重复点，重复的点只留下一个，
 * 并且还要修改观测关系
 * @param curr_match_lms_observed_in_cand 回环关键帧以及共视关键帧观测到的所有地图点
 * @param Sim3s_nw_after_correction 回环修正之后的当前关键帧以及共视关键帧
 */
void global_optimization_module::replace_duplicated_landmarks(const std::vector<data::landmark*>& curr_match_lms_observed_in_cand,
                                                              const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    // resolve duplications of landmarks between the current keyframe and the loop candidate
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        //首先处理那些已知的匹配上的观测，直接将回环帧的地图点替换掉当前帧的地图点
        for (unsigned int idx = 0; idx < cur_keyfrm_->num_keypts_; ++idx) {
            auto curr_match_lm_in_cand = curr_match_lms_observed_in_cand.at(idx);
            if (!curr_match_lm_in_cand) {//跳过那些没有匹配上
                continue;
            }

            auto lm_in_curr = cur_keyfrm_->get_landmark(idx);//此时的landmark是已经经过回环修正的
            if (lm_in_curr) {
                // if the landmark corresponding `idx` exists,
                // replace it with `curr_match_lm_in_cand` (observed in the candidate)
                lm_in_curr->replace(curr_match_lm_in_cand);
            }
            else {
                // if landmark corresponding `idx` does not exists,
                // add association between the current keyframe and `curr_match_lm_in_cand`
                cur_keyfrm_->add_landmark(curr_match_lm_in_cand, idx);
                curr_match_lm_in_cand->add_observation(cur_keyfrm_, idx);
                curr_match_lm_in_cand->compute_descriptor();
            }
        }
    }

    // resolve duplications of landmarks between the current keyframe and the candidates of the loop candidate

    //回环帧对应的共视关键帧的所有地图点
    const auto curr_match_lms_observed_in_cand_covis = loop_detector_->current_matched_landmarks_observed_in_candidate_covisibilities();
    match::fuse fuser(0.8);
    for (const auto& t : Sim3s_nw_after_correction) {//回环修正之后的当前关键帧以及共视关键帧进行循环
        auto neighbor = t.first;
        const Mat44_t Sim3_nw_after_correction = util::converter::to_eigen_mat(t.second);

        // reproject the landmarks observed in the current keyframe to the neighbor,
        // then search duplication of the landmarks
        std::vector<data::landmark*> lms_to_replace(curr_match_lms_observed_in_cand_covis.size(), nullptr);
        fuser.detect_duplication(neighbor, Sim3_nw_after_correction, curr_match_lms_observed_in_cand_covis, 4, lms_to_replace);

        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
        // if any landmark duplication is found, replace it
        for (unsigned int i = 0; i < curr_match_lms_observed_in_cand_covis.size(); ++i) {
            auto lm_to_replace = lms_to_replace.at(i);
            if (lm_to_replace) {
                lm_to_replace->replace(curr_match_lms_observed_in_cand_covis.at(i));
            }
        }
    }
}

/*!
 * 主要解决当前关键帧合并到回环关键帧之后的，关键帧去留问题，这里的策略是删掉当前关键帧以及回环关键帧，留下与回环候选关键帧相关的那些关键帧
 * 留下回环关键帧以及它的共视关键帧
 * @param covisibilities
 * @return 返回包含回环帧以及候选它的共视关键帧
 */
auto global_optimization_module::extract_new_connections(const std::vector<data::keyframe*>& covisibilities) const
    -> std::map<data::keyframe*, std::set<data::keyframe*>> {
    std::map<data::keyframe*, std::set<data::keyframe*>> new_connections;

    for (auto covisibility : covisibilities) {//对当前关键帧以及共视关键帧进行循环
        // acquire neighbors BEFORE loop fusion (because update_connections() is not called yet)
        const auto neighbors_before_update = covisibility->graph_node_->get_covisibilities();//当前关键帧的每一个共视关键帧的共视关键帧

        // call update_connections()
        covisibility->graph_node_->update_connections();//由于之前进行了地图点合并，所以共视关系发生了改变，需要更新共视关系
        // acquire neighbors AFTER loop fusion
        new_connections[covisibility] = covisibility->graph_node_->get_connected_keyframes();//记录更新共视关系之后的的当前关键帧与共视关键帧

        // remove covisibilities
        for (const auto keyfrm_to_erase : covisibilities) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);//更新连接关系之后，删除当前关键帧以及共视关键帧，相当于合并之后留下的是回环帧以及回环帧的共视关键帧，将它们作为当前的关键帧
        }
        // remove nighbors before loop fusion
        for (const auto keyfrm_to_erase : neighbors_before_update) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);//删除当前关键帧的每一个共视关键帧的共视关键帧
        }
    }

    return new_connections;
}

void global_optimization_module::request_reset() {
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        reset_is_requested_ = true;
    }

    // BLOCK until reset
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx_reset_);
            if (!reset_is_requested_) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

bool global_optimization_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void global_optimization_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset global optimization module");
    keyfrms_queue_.clear();
    loop_detector_->set_loop_correct_keyframe_id(0);
    reset_is_requested_ = false;
}

void global_optimization_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

bool global_optimization_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool global_optimization_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void global_optimization_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause global optimization module");
    is_paused_ = true;
}

void global_optimization_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume global optimization module");
}

void global_optimization_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool global_optimization_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool global_optimization_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void global_optimization_module::terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    is_terminated_ = true;
}

bool global_optimization_module::loop_BA_is_running() const {
    return loop_bundle_adjuster_->is_running();
}

void global_optimization_module::abort_loop_BA() {
    loop_bundle_adjuster_->abort();
}

} // namespace openvslam
