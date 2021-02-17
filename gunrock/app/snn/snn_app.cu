// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file snn_app.cu
 *
 * @brief Simple Gunrock Application
 */

#include <cstdio>
#include <iostream>

// Gunrock api
#include <gunrock/gunrock.h>

// Test utils
#include <gunrock/util/test_utils.cuh>

// Graphio include
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/graphio/labels.cuh>

// App and test base includes
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// JSON includes
#include <gunrock/util/info_rapidjson.cuh>

// SNN includes
#include <gunrock/app/snn/snn_enactor.cuh>
#include <gunrock/app/snn/snn_test.cuh>

// FAISS knn
#ifdef FAISS_FOUND
    #include <faiss/gpu/GpuDistance.h>
    #include <faiss/gpu/GpuIndexFlat.h>
    #include <faiss/gpu/GpuResources.h>
    #include <faiss/gpu/StandardGpuResources.h>
    #include <faiss/utils/Heap.h>
    #include <faiss/gpu/utils/Limits.cuh>
    #include <faiss/gpu/utils/Select.cuh>
#endif

#define SNN_APP_DEBUG 1
#ifdef SNN_APP_DEBUG
    int print_max_n = 20;
    int print_max_k = 5;
    int print_max_dim = 5;
    #define debug(a...) printf(a)
#else
    #define debug(a...) 
#endif


namespace gunrock {
namespace app {
namespace snn {

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(UseParameters_app(parameters));
  GUARD_CU(UseParameters_problem(parameters));
  GUARD_CU(UseParameters_enactor(parameters));

  GUARD_CU(parameters.Use<std::string>(
      "labels-file",
      util::REQUIRED_ARGUMENT | util::REQUIRED_PARAMETER, 
      "", "List of points of dim-dimensional space", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "transpose",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "False if lables are not transpose", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "knn-version",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "faiss", "Version of knn: \"gunrock\" or \"kmcuda\" or \"cuml\" or \"faiss\"", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "snn-tag", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, "",
      "snn-tag info for json string", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "n",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Numbers of points", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "dim",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Dimension of labels", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "k",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      10, "Numbers of k neighbors.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "eps",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "The minimum number of neighbors two points should share\n"
      "to be considered close to each other",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<uint32_t>(
      "min-pts",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "The minimum density that a point should have to be considered a core "
      "point\n",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "NUM-THREADS",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      128, "Number of threads running per block.", __FILE__, __LINE__));
 
  GUARD_CU(parameters.Use<bool>(
      "use-shared-mem",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "True if kernel must use shared memory.", __FILE__, __LINE__));
 
  GUARD_CU(parameters.Use<float>(
      "cpu-elapsed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
      "CPU implementation, elapsed time (ms) for JSON.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "knn-elapsed", util::REQUIRED_ARGUMENT | util::OPTIONAL_PARAMETER, 0.0f,
      "KNN Gunrock implementation, elapsed time (ms) for JSON.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "save-snn-results",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Save cluster assignments to file", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "snn-output-file",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "snn_results.output", "Filename of snn output", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Run knn tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT, typename SizeT = typename GraphT::SizeT>
cudaError_t RunTests(
    util::Parameters &parameters, GraphT &graph, SizeT num_points, SizeT k,
    SizeT eps, SizeT min_pts,
    SizeT *h_knns,
    SizeT *h_cluster, SizeT *ref_cluster,
    SizeT *h_core_point_counter, SizeT *ref_core_point_counter,
    SizeT *h_noise_point_counter, SizeT *ref_noise_point_counter, 
    SizeT *h_cluster_counter, SizeT *ref_cluster_counter, 
    util::Location target) {
  cudaError_t retval = cudaSuccess;

  //typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  int num_runs = parameters.Get<int>("num-runs");
  bool save_snn_results = parameters.Get<bool>("save-snn-results");
  std::string snn_output_file = parameters.Get<std::string>("snn-output-file");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("snn", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(h_knns, target));
    GUARD_CU(enactor.Reset(target));

    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    if (validation == "each" && 
	ref_cluster && ref_core_point_counter && ref_noise_point_counter && ref_cluster_counter) {
      GUARD_CU(problem.Extract(num_points, k, h_cluster, h_core_point_counter,
                  h_noise_point_counter, h_cluster_counter));
      SizeT num_errors = Validate_Results(parameters, graph, h_cluster,
                               h_core_point_counter, h_noise_point_counter, 
                               h_cluster_counter,
                               ref_cluster, ref_core_point_counter, 
                               ref_noise_point_counter, 
                               ref_cluster_counter, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(num_points, k, h_cluster, h_core_point_counter, 
              h_noise_point_counter, h_cluster_counter));
  if (validation == "last" &&
      ref_cluster && ref_core_point_counter && ref_noise_point_counter && ref_cluster_counter) {
    SizeT num_errors = Validate_Results(parameters, graph, h_cluster,
                                h_core_point_counter, h_noise_point_counter, 
                                h_cluster_counter,
                                ref_cluster, ref_core_point_counter, 
                                ref_noise_point_counter, 
                                ref_cluster_counter, false);
  }

  // compute running statistics
  // Change NULL to problem specific per-vertex visited marker, e.g.
  // h_distances
  info.ComputeTraversalStats(enactor, (SizeT *)NULL);
  // Display_Memory_Usage(problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
  // Display_Performance_Profiling(enactor);
#endif

  // For JSON output
  info.SetVal("num-corepoints", std::to_string(h_core_point_counter[0]));
  info.SetVal("num-noisepoints", std::to_string(h_noise_point_counter[0]));
  info.SetVal("num-clusters", std::to_string(h_cluster_counter[0]));
  // info.SetVal("cpu-elapsed",
  // std::to_string(parameters.Get<float>("cpu-elapsed")));
  if (save_snn_results){
    std::ofstream output(snn_output_file);
    for (int i=0; i<num_points; ++i){
        output << h_cluster[i] << "\n";
    }
    output.close();
  }

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

template <typename ArrayT, typename SizeT = int, typename ValueT = double>
cudaError_t RunKNN(util::Parameters& parameters, const SizeT knn_version,
    const SizeT num_points, const SizeT dim, const SizeT k, 
    const ArrayT& points, SizeT* h_knns){ 
    
    gunrock::util::PrintMsg("KNN version: " + knn_version);
    bool quiet = parameters.Get<bool>("quiet");
    cudaError_t retval = cudaSuccess;

    if (knn_version == 0){//GUNROCK
    }else if (knn_version == 1){//FAISS

#ifdef FAISS_FOUND
        //* -------------------- FAISS KNN ------------------------*
        long* res_I;
        GUARD_CU(cudaMalloc((void**)&res_I, sizeof(long)*num_points*(k+1)));
        float* res_D;
        GUARD_CU(cudaMalloc((void**)&res_D, sizeof(float)*num_points*(k+1)));

        ValueT *samples0 = (ValueT*)points.GetPointer(gunrock::util::HOST);
        float *samples = (float*)malloc(num_points * dim * sizeof(float));

        for (int i = 0; i < num_points * dim; ++i) 
            samples[i] = (float)samples0[i];
        
        std::vector<float*> ptrs(1);
        ptrs[0] = samples;
        std::vector<int> sizes(1);
        sizes[0] = num_points;

        SizeT device = parameters.Get<SizeT>("device");
        GUARD_CU(cudaSetDevice(device));
        cudaStream_t stream;
        GUARD_CU(cudaStreamCreate(&stream));

        faiss::gpu::StandardGpuResources gpu_res;
        gpu_res.noTempMemory();
        gpu_res.setDefaultStream(device, stream);
        gunrock::util::CpuTimer cpu_timer;
        cpu_timer.Start();
        faiss::gpu::bruteForceKnn(&gpu_res, faiss::METRIC_L2, samples, true, num_points,
                samples, true, num_points, dim, k+1, res_D, res_I);
        cpu_timer.Stop();
    
        gunrock::util::PrintMsg("Faiss KNN Elapsed: " 
                  + std::to_string(cpu_timer.ElapsedMillis()), !quiet);
        gunrock::util::PrintMsg("__________________________", !quiet);
        parameters.Set("knn-elapsed", cpu_timer.ElapsedMillis());
    
        long* knn_res = (long*)malloc(sizeof(long)*num_points*(k+1));
        GUARD_CU(cudaMemcpy(knn_res, res_I, sizeof(long)*num_points*(k+1), cudaMemcpyDeviceToHost));
        GUARD_CU(cudaDeviceSynchronize());

        for (SizeT x = 0; x < num_points; ++x){
            if (knn_res[x * (k+1)] != x){
                h_knns[x*k] = knn_res[x * (k+1)];
            }
            for (int i=0; i<k; ++i){
                if (knn_res[x * (k+1) + i + 1] == x)
                    continue;
                h_knns[x*k + i] = knn_res[x * (k+1) + i + 1];
            }
        }
        delete [] samples;
        delete [] knn_res;
        cudaFree(res_I);
        cudaFree(res_D);
#else 
        // FAISS_FOUND
        gunrock::util::PrintMsg("FAISS library not found.");
        printf("FAISS library not found.");
        delete [] h_knns;
#endif 

    } // if FAISS
    return retval;
} // RunKNN

}  // namespace snn
}  // namespace app
}  // namespace gunrock


template <typename ValueT = double, typename SizeT = int, typename VertexT = int>
cudaError_t snn(const std::string labels, const SizeT k, 
            const SizeT eps, const SizeT min_pts, SizeT *h_cluster, 
            SizeT *h_cluster_counter, SizeT *h_core_point_counter, 
            SizeT *h_noise_point_counter, SizeT knn_version=1){
    
    // Setup parameters
    gunrock::util::Parameters parameters("snn");
    gunrock::graphio::UseParameters(parameters);
    gunrock::app::snn::UseParameters(parameters);
    gunrock::app::UseParameters_test(parameters);
    parameters.Parse_CommandLine(0, NULL);
    parameters.Set("labels-file", labels);
    parameters.Set("k", k);
    parameters.Set("eps", eps);
    parameters.Set("min-pts", min_pts);

    // Creating points array from labels
    gunrock::util::Array1D<SizeT, ValueT> points;
    cudaError_t retval = cudaSuccess;
    retval = gunrock::graphio::labels::Read(parameters, points);
    if (retval){
        gunrock::util::PrintMsg("Reading error");
        return retval;
    }
   
    // Check the input labels
    SizeT num_points = parameters.Get<SizeT>("n");
    SizeT dim = parameters.Get<SizeT>("dim");

#ifdef SNN_APP_DEBUG
    for (int i=0; i<(num_points < print_max_n ? num_points : print_max_n); ++i){
        debug("%d: ", i);
        for (int j=0; j<(dim < print_max_dim ? dim : print_max_dim); ++j){
            debug("%lf ", points[i*dim +j]);
        }
        debug("... \n");
    }
#endif

    if (k >= num_points){
        printf("k = %d > num_points %d, dones\n", k, num_points);
        return gunrock::util::GRError("k must be smaller than the number of labels", __FILE__, __LINE__);}

    // Run KNN
    SizeT* h_knns = (SizeT*) malloc(sizeof(SizeT)*num_points*k);
    GUARD_CU(gunrock::app::snn::RunKNN(parameters, knn_version, num_points, 
        dim, k, points, h_knns));

#ifdef SNN_APP_DEBUG
    debug("Run KNN results: \n");
        for (int i=0; i<(num_points < print_max_n ? num_points : print_max_n); ++i){
        debug("%d: ", i);
        for (int j=0; j<(k < print_max_k ? k : print_max_k); ++j){
            debug("%d ", h_knns[i*k +j]);
        }
        debug("\n");
    }
#endif

    // Run SNN
    typedef typename gunrock::app::TestGraph<
        VertexT, SizeT, ValueT, gunrock::graph::HAS_CSR> GraphT;
    GraphT graph;
    // Result on GPU
    /*
    h_cluster = (SizeT*)malloc(sizeof(SizeT) * num_points);
    h_core_point_counter = (SizeT*)malloc(sizeof(SizeT));
    h_noise_point_counter = (SizeT*)malloc(sizeof(SizeT));
    h_cluster_counter = (SizeT*)malloc(sizeof(SizeT));
    */

    GUARD_CU(gunrock::app::snn::RunTests(parameters, graph, num_points, k, 
                                eps, min_pts, h_knns, h_cluster, (SizeT*)NULL,
                                h_core_point_counter, (SizeT*)NULL,
                                h_noise_point_counter, (SizeT*)NULL,
                                h_cluster_counter, (SizeT*)NULL, 
                                gunrock::util::DEVICE));

#ifdef SNN_APP_DEBUG
    debug("Core Points: %d\n", *h_core_point_counter);
    debug("Noise Points: %d\n", *h_noise_point_counter);
    debug("Cluster Counter: %d\n", *h_cluster_counter);
    for (int i=0; i<(num_points < print_max_n ? num_points : print_max_n); ++i){
		debug("cluster[%d] = %d\n", i, h_cluster[i]);
    }
#endif

    return retval;
}

/*
 * @brief SNN interface 
 * @param[in]   labels_file          The filepath of labels
 * @param[in]   k                    The kNN parameter
 * @param[in]   eps                  Parameter of density
 * @param[in]   min_pts              Parameter of core point
 * @param[out]  clusters             Return cluster assignments
 * @param[out]  clusters_counter     Return the number of clusters
 * @param[out]  core_points_counter  Return the number of core points
 * @param[out]  noise_points_counter Return the number of noise points
 * \return      double               Return accumulated elapsed times for all runs
 */
double snn(const char* labels_file, const int* k, const int* eps, 
            const int* min_pts, int *clusters, 
            int *clusters_counter, int *core_points_counter, 
            int *noise_points_counter){
    std::string labels(labels_file);
    int K = *k;
    int Eps = *eps;
    int Min_pts = *min_pts;
    gunrock::util::CpuTimer cpu_timer;
    cpu_timer.Start();
    cudaError_t retval = snn(labels, K, Eps, Min_pts, clusters, clusters_counter, 
		    core_points_counter, noise_points_counter, 1);
    if (retval){
	gunrock::util::GRError("gunrock.snn returns error");
    }
    cpu_timer.Stop();
    auto elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
