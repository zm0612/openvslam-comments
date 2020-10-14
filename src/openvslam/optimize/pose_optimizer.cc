#include "openvslam/data/frame.h"
#include "openvslam/data/landmark.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/optimize/g2o/se3/pose_opt_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <vector>
#include <mutex>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace openvslam {
namespace optimize {

pose_optimizer::pose_optimizer(const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer::optimize(data::frame& frm) const {

    // 1. 设置g2o优化器，这些都是固定套路，选择线性求解器、块求解器，然后选择迭代方法
    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverEigen<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));//块求解器中，顶点的维度为6，误差项的维度为3
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));//std::move将左值转换为右值，实际是转换所有权

    // 选择稀疏优化器，并且设置优化方法
    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // 用于记录在优化前所有的观测，相当于是初始优化时的边的数量
    unsigned int num_init_obs = 0;

    // 2. 设置图优化的顶点，因为这里优化的是相机的pose，所以顶点就一个
    auto frm_vtx = new g2o::se3::shot_vertex();//使用g2o提供的顶点类型
    frm_vtx->setId(frm.id_);
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(frm.cam_pose_cw_));//将SE3转换为g2o表示形式：四元素和平移向量
    frm_vtx->setFixed(false);//必须设为不固定，不然优化时顶点永远不会变化
    optimizer.addVertex(frm_vtx);

    const unsigned int num_keypts = frm.num_keypts_;

    // 3. 设置图优化的边，这里的边是重投影误差
    using pose_opt_edge_wrapper = g2o::se3::pose_opt_edge_wrapper<data::frame>;

    std::vector<pose_opt_edge_wrapper> pose_opt_edge_wraps;
    pose_opt_edge_wraps.reserve(num_keypts);

    //卡方校验(chi-square test)，在自由度为2，并且置信度为95%
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);

    //卡方校验(chi-square test)，在自由度为3，并且置信度为95%
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    // 设置每一条边
    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto lm = frm.landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        ++num_init_obs;
        frm.outlier_flags_.at(idx) = false;

        // frameのvertexをreprojection edgeで接続する
        const auto& undist_keypt = frm.undist_keypts_.at(idx);
        const float x_right = frm.stereo_x_right_.at(idx);
        const float inv_sigma_sq = frm.inv_level_sigma_sq_.at(undist_keypt.octave);
        const auto sqrt_chi_sq = (frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;

        //设置边
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(&frm, frm_vtx, lm->get_pos_in_world(),
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
        pose_opt_edge_wraps.push_back(pose_opt_edge_wrap);
        optimizer.addEdge(pose_opt_edge_wrap.edge_);//增加边
    }

    //边少于5个就没有必要优化了
    if (num_init_obs < 5) {
        return 0;
    }

    // 4. robust BAを実行する

    unsigned int num_bad_obs = 0;//用于记录优化过程中，通过卡方值判断位outlier的边
    for (unsigned int trial = 0; trial < num_trials_; ++trial) {
        optimizer.initializeOptimization();

        //开始优化
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            auto edge = pose_opt_edge_wrap.edge_;

            if (frm.outlier_flags_.at(pose_opt_edge_wrap.idx_)) {
                edge->computeError();
            }

            //如果有边不满足卡方校验，那么就将其设置为outlier，然后继续重新优化
            if (pose_opt_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2()) {//用卡方值进行outlier值判断
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2()) {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }

            //当优化回数剩下两次时，就不要在设置鲁棒核函数了
            if (trial == num_trials_ - 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        //如果优化之后剩下的边小于5，那就不用再优化了
        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    // 5. 情報を更新

    frm.set_cam_pose(frm_vtx->estimate());//更新frame的pose

    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace openvslam
