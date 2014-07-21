// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * topk_app.cu
 *
 * @brief topk app implementation
 */

#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <gunrock/gunrock.h>
#include <gunrock/graphio/market.cuh>
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::topk;

/*
 * @brief searches for a value in sorted array
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] arr is an array to search in
 * @param[in] val is searched value
 * @param[in] left  is an index of left  boundary
 * @param[in] right is an index of right boundary
 *
 * return the searched value, if it presents in the array
 * return -1 if the searched value is absent
 */
template <
    typename VertexId,
    typename SizeT>
int binary_search(
    VertexId *arr,
    VertexId val,
    SizeT    left,
    SizeT    right)
{
    while (left <= right) {
	int mid = (left + right) / 2;
	if (arr[mid] == val)
	    return arr[mid];
	else if (arr[mid] > val)
	    right = mid - 1;
	else
	    left = mid + 1;
    }
    return -1;
}


/**
 * @brief Build SubGraph Contains Only Top K Nodes
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[out] output subgraph of topk problem
 * @param[in]  input graph need to process on
 * @param[in]  topk node_ids
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void build_topk_subgraph(
    GunrockGraph *subgraph,
    const Csr<VertexId, Value, SizeT> &graph_original,
    const Csr<VertexId, Value, SizeT> &graph_reversed,
    VertexId  *node_ids,
    int	      top_nodes)
{
    int search_return = 0;
    int search_count  = 0;
    std::vector<VertexId> node_ids_vec(node_ids, node_ids + top_nodes);
    std::vector<int>      sub_row_offsets;
    std::vector<VertexId> sub_col_indices;

    // build row_offsets and col_indices of sub-graph
    sub_row_offsets.push_back(0); // start of row_offsets
    for (int i = 0; i < top_nodes; ++i) {
	for (int j = 0; j < top_nodes; ++j) {
	    /*
	    // debug print
	    printf("searching %d in column_indices[%d, %d) = [", node_ids[j],
	    graph_original.row_offsets[node_ids[i]],
	    graph_original.row_offsets[node_ids[i]+1]);
	    for (int k = graph_original.row_offsets[node_ids[i]];
	    k < graph_original.row_offsets[node_ids[i]+1]; ++k)
	    {
	    printf(" %d", graph_original.column_indices[k]);
	    }
	    printf("]\n");
	    */

	    search_return = binary_search<VertexId, SizeT>(
		graph_original.column_indices,
		node_ids[j],
		graph_original.row_offsets[node_ids[i]],    // [left
		graph_original.row_offsets[node_ids[i]+1]); // right)
	    // filter col_indices
	    if (search_return != -1) {
		++search_count;
		// TODO: improve efficiency
		search_return = std::find(
		    node_ids_vec.begin(),
		    node_ids_vec.end(),
		    search_return) - node_ids_vec.begin();
		sub_col_indices.push_back(search_return);
	    }
	}
	// build sub_row_offsets
	search_count += sub_row_offsets[sub_row_offsets.size()-1];
	sub_row_offsets.push_back(search_count);
	search_count = 0;
    }

    // generate subgraph of top k nodes
    subgraph->num_nodes	= top_nodes;
    subgraph->num_edges	= sub_col_indices.size();
    subgraph->row_offsets = &sub_row_offsets[0];
    subgraph->col_indices = &sub_col_indices[0];

    /*
    // display sub-graph
    Csr<int, int, int> test_graph(false);
    test_graph.nodes = subgraph->num_nodes;
    test_graph.edges = subgraph->num_edges;
    test_graph.row_offsets    = (int*)subgraph->row_offsets;
    test_graph.column_indices = (int*)subgraph->col_indices;

    test_graph.DisplayGraph();

    test_graph.row_offsets    = NULL;
    test_graph.column_indices = NULL;
    */

    // clean up
    node_ids_vec.clear();
    sub_row_offsets.clear();
    sub_col_indices.clear();
}

/**
 * @brief Run TopK
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] output subgraph of topk problem
 * @param[out] node_ids return the top k nodes
 * @param[out] centrality_values return associated centrality
 * @param[in]  original graph to the CSR graph we process on
 * @param[in]  reversed graph to the CSR graph we process on
 * @param[in]  top_nodes k value for topk problem
 *
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void topk_run(
    GunrockGraph *graph_out,
    VertexId	 *node_ids,
    Value	 *centrality_values,
    const Csr<VertexId, Value, SizeT> &graph_original,
    const Csr<VertexId, Value, SizeT> &graph_reversed,
    SizeT top_nodes)
{
    // preparations
    typedef TOPKProblem<VertexId, SizeT, Value> Problem;
    TOPKEnactor<false> topk_enactor(false);
    Problem *topk_problem = new Problem;

    // reset top_nodes if necessary
    top_nodes =
	(top_nodes > graph_original.nodes) ? graph_original.nodes : top_nodes;

    // initialization
    util::GRError(topk_problem->Init(
		      false,
		      graph_original,
		      graph_reversed,
		      1),
		  "Problem TOPK Initialization Failed", __FILE__, __LINE__);

    // reset data slices
    util::GRError(topk_problem->Reset(topk_enactor.GetFrontierType()),
		  "TOPK Problem Data Reset Failed", __FILE__, __LINE__);

    // launch gpu topk enactor to calculate top k nodes
    util::GRError(topk_enactor.template Enact<Problem>(
		      topk_problem,
		      top_nodes),
		  "TOPK Problem Enact Failed", __FILE__, __LINE__);

    // copy out results back to cpu
    util::GRError(topk_problem->Extract(
		      node_ids,
		      centrality_values,
		      top_nodes),
		  "TOPK Problem Data Extraction Failed", __FILE__, __LINE__);

    // build a subgraph contains only top k nodes on cpu
    build_topk_subgraph<VertexId, Value, SizeT>(
	graph_out,
	graph_original,
	graph_reversed,
	(int*)node_ids,
	top_nodes);

    // cleanup if neccessary
    if (topk_problem) { delete topk_problem; }

    cudaDeviceSynchronize();
}

/*
 * @brief topk dispatch function base on gunrock data types
 *
 * @param[out] output subgraph of topk problem
 * @param[out] output top k node_ids
 * @param[out] output associated centrality values
 * @param[in]  input graph need to process on
 * @param[in]  k value of topk problem
 * @param[in]  gunrock datatype struct
 */
void gunrock_topk(
    GunrockGraph       *graph_out,
    void               *node_ids,
    void               *centrality_values,
    const GunrockGraph *graph_in,
    GunrockConfig	   topk_config,
    GunrockDataType    data_type)
{
    //TODO: add more supportive datatypes if necessary

    switch (data_type.VTXID_TYPE)
      {
      case VTXID_INT:
	switch(data_type.SIZET_TYPE)
	{
	case SIZET_UINT:
	  switch (data_type.VALUE_TYPE)
	    {
	    case VALUE_INT:
	      {
		// case that VertexId, SizeT, Value are all have the type int

		// original graph
		Csr<int, int, int> graph_original(false);
		graph_original.nodes = graph_in->num_nodes;
		graph_original.edges = graph_in->num_edges;
		graph_original.row_offsets    = (int*)graph_in->row_offsets;
		graph_original.column_indices = (int*)graph_in->col_indices;

		// reversed graph
		Csr<int, int, int> graph_reversed(false);
		graph_reversed.nodes = graph_in->num_nodes;
		graph_reversed.edges = graph_in->num_edges;
		graph_reversed.row_offsets    = (int*)graph_in->col_offsets;
		graph_reversed.column_indices = (int*)graph_in->row_indices;

		graph_original.DisplayGraph();

		topk_run<int, int, int>(
		    graph_out,
		    (int*)node_ids,
		    (int*)centrality_values,
		    graph_original,
		    graph_reversed,
		    topk_config.top_nodes);

		// reset for free memory
		graph_original.row_offsets    = NULL;
		graph_original.column_indices = NULL;
		graph_reversed.row_offsets    = NULL;
		graph_reversed.column_indices = NULL;

		break;
	      }
	    case VALUE_FLOAT:
	      {
		// case that VertexId and SizeT have type int, Value is float
		/*
		// original graph
		Csr<int, float, int> graph_original(false);
		graph_original.nodes = graph_in->num_nodes;
		graph_original.edges = graph_in->num_edges;
		graph_original.row_offsets    = (int*)graph_in->row_offsets;
		graph_original.column_indices = (int*)graph_in->col_indices;

		// reversed graph
		Csr<int, float, int> graph_reversed(false);
		graph_reversed.nodes = graph_in->num_nodes;
		graph_reversed.edges = graph_in->num_edges;
		graph_reversed.row_offsets    = (int*)graph_in->col_offsets;
		graph_reversed.column_indices = (int*)graph_in->row_indices;

		topk_run<int, float, int>((int*)node_ids,
					  (float*)centrality_values,
					  graph_original,
					  graph_reversed,
					  top_nodes);

		// reset for free memory
		graph_original.row_offsets    = NULL;
		graph_original.column_indices = NULL;
		graph_reversed.row_offsets    = NULL;
		graph_reversed.column_indices = NULL;
		*/
		break;
	      }
	    }
	  break;
	}
      break;
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: