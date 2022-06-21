/**
 * This file includes the C++ side of communicating functions between C++ and Python. CTypes is used for the communication.
 * */

#include "graph.h"
#include <cstring>
#include <iostream>
using namespace std;

Graph graphs; 									// holds batch many graphs 
std::vector<std::vector<float>> node_embeds; 	// holds each node embedding for each batch
std::vector<float> graph_embeds; 				// holds each graph embedding for each batch
std::vector<int> color_arrs;					// for each batch, coloring information of all nodes are stored

int nfeatures_size = 9;										// node embedding related feature size
int gfeatures_size = 15;									// graph embedding related feature size

/**
 * Gets batch and a node interval as input. Creates batch many graphs having node count in the given interval and
 * edge count between: 10*num_nodes and (num_nodes)(num_nodes - 1) / 4
 * */
extern "C" int insert_batch(int min_nodes, int max_nodes)
{
	srand(112);
	int node_cnts = 0
	// Initialization of embedding and coloring vectors with the given sizes as input
	
	node_embeds = std::vector<std::vector<float>>(std::vector<float>(nfeatures_size)); //여기까지
	graph_embeds = std::vector<float>
	color_arrs = std::vector<int>;

	int n = rand() % (max_nodes - min_nodes) + min_nodes; 	// vertex count is determined
	int max_edges = n*(n-1)/4, min_edges = n;				// edge interval is set
	if(n > 20) min_edges = 10 * n;
	int e = rand() % (max_edges - min_edges) + min_edges;	// edge count is determined
	Graph g(n, e);											// graph with determined edge and vertex count is created , randomly
	graphs = g;

	node_cnts = n;
	for(int k = 0; k < nfeatures_size; k++) {
		node_embeds[k] = std::vector<float>(n, 0);
	}
	color_arrs = std::vector<int> (n, -1);

	return node_cnts;
}



/**
 * Batch vector for graphs is reset
 * */
extern "C" void reset_batch()
{
	graphs.clear();
	graphs.resize(0);
}


/**
 * Standard normalization function for single ordering vector
 * */
void normal_params(std::vector<float> &order, float &mean, float &stdev)
{
    float sum = 0;
    float sq_sum = 0;
     auto &x : order)
    {
        sum += x;
        sq_sum += x * x;
    }
    mean = sum / order.size();
    stdev = sqrt((sq_sum / order.size()) - (mean * mean));
    if (std::isnan(stdev) || stdev == 0)
        stdev = 0.001;
}

/**
 * Standard normalization function for collection of ordering vectors.
 * */
void normalize_node_embed(std::vector<std::vector<float>> &orders)
{
    float mean, stdev;
    for (unsigned int i = 0; i < orders.size(); i++)
    {
        normal_params(orders[i], mean, stdev);
        for_each(orders[i].begin(), orders[i].end(), [mean, stdev](float &x) { x = (x - mean) / stdev; });
    }
}

/**
 * Graph embedding is initialized for the graph in the given index in batch.
 * */
void init_graph_embed()
{
	auto curr_graph = graphs;
	auto &row_ptr = curr_graph.row_ptr;
	int size = curr_graph.num_nodes;
	
	vector<float> closeness;
	curr_graph.closeness_centrality_approx(closeness);

	graph_embeds = std::vector<float>(gfeatures_size, 0);
	auto graph_embed = graph_embeds;

	graph_embed[0] = (float)curr_graph.num_edges / size; // num edges / num nodes

	graph_embed[1] = 0; // nof adjacents of colored nodes
	graph_embed[2] = 0; // nof colored adjacents of colored nodes
	graph_embed[3] = 0; // closeness sum of colored nodes
	graph_embed[4] = 0; // nof colored nodes
	for (int i = 0; i < (int)closeness.size(); i++)
	{ // closeness sum of uncolored nodes
		graph_embed[5] += (float)closeness[i];
	}
	graph_embed[6] = size; // nof uncolored nodes
	graph_embed[7] = 0;	   // degree above mean (among colored nodes)
	graph_embed[8] = 0;	   // degree below mean (among colored nodes)
	graph_embed[9] = 0;	   // degree above mean (among uncolored nodes)
	graph_embed[10] = 0;   // degree below mean (among uncolored nodes)
	double degree_mean = (double)row_ptr.back() / size;
	double degree_sq_sum = 0;
	for (int v = 0; v < (int)row_ptr.size() - 1; v++)
	{
		int degree = row_ptr[v + 1] - row_ptr[v];
		if (degree > degree_mean)
		{
			graph_embed[9]++;
		}
		else
		{
			graph_embed[10]++;
		}
		degree_sq_sum += degree * degree;
	}
	double variance = degree_sq_sum / size - (degree_mean * degree_mean);
	graph_embed[11] = 0;		   // mean degree of colored nodes
	graph_embed[12] = degree_mean; // mean degree of uncolored nodes
	graph_embed[13] = 0;		   // variance of degrees (colored nodes)
	graph_embed[14] = variance;	   // variance of degrees (uncolored nodes)
}


/**
 * Graph embeddings of given batch is initialized
 * */
extern "C" void init_graph_embeddings()
{
	init_graph_embed(i);
}

/**
 * Graph embedding for the selected graph in the batch is returned
 * */
extern "C" float *get_graph_embed(int *size)
{
	*size = graph_embeds.size();
	return graph_embeds.data();
}


/**
 * Taking transpose of 2D matrix, for our case transpose of node embedding of a
 * single graph in batch
 * */
void transpose(vector<vector<float>> &node_embeddings)
{
	int rows = (int)node_embeddings.size();
	int cols = (int)node_embeddings[0].size();
	vector<vector<float>> res(cols, vector<float>(rows));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			res[j][i] = node_embeddings[i][j];
		}
	}
	node_embeddings = res;
}

/**
 * Node embeddings for the selected graph in batch is calculated initially
 * and afterwards, transpose of the result is taken for easier reachability in python side
 * */
void init_node_embed()
{
	auto curr_graph = graphs;
	curr_graph.degree_order(node_embeds[0]);
	curr_graph.degree_2_order(node_embeds[1]);
	curr_graph.degree_3_order(node_embeds[2]);
	curr_graph.closeness_centrality_approx(node_embeds[3]);
	curr_graph.clustering_coeff(node_embeds[4]);
	curr_graph.page_rank(node_embeds[5]);
	// dynamic coeffs
	// 6th index -> nof colored neighbors
	// 7th index -> nof different colored neighbors
	// 8th index -> is node itself colored
	// dynamic coefficients are automatically 0 in the beginning
	normalize_node_embed(node_embeds);
	transpose(node_embeds);
}


/**
 * For each batch, node embeddings are initialized
 * */
extern "C" void init_node_embeddings()
{
	for (unsigned int i = 0; i < graphs.size(); i++)
	{
		init_node_embed(i);
	}
}

/**
 * For single graph in batch, node embeddings are returned
 * Row and Col values stands for the sizes of node embedding matrix
 * */
extern "C" float **get_node_embed(int *row, int *col)
{
	*row = node_embeds.size();
	*col = node_embeds[0].size();
	float **res = new float *[*row];
	for (int i = 0; i < *row; i++)
	{
		res[i] = node_embeds[i].data(); // copying the embedding data to 2D pointer
	}
	return res;
}

/**
 * Updating node embedding for single graph in the batch.
 * */
void update_node_embed(int node, int color)
{
	auto curr_graph = graphs;
	auto node_embed = node_embeds;
	for (int edge = curr_graph.row_ptr[node]; edge < curr_graph.row_ptr[node + 1]; edge++)
	{ // each adjacent of the colored node is updated
		int adj = curr_graph.col_ind[edge];
		if(color_arrs[adj] == -1) {
			// number of colored neighbors are increased by one (division is for normalization of value)
			node_embed[6] += (float) 1 / curr_graph.num_nodes;
		}

		if (color > node_embed[adj][7])
		{
			// if the color selected is bigger then adj's all the neighboring nodes' colors, then this color is
			// set to adj's embedding (division to max degree is for normalization)
			if(color_arrs[adj] == -1 && node_embed[adj][7] < ((float) color / curr_graph.max_degree)) {
				node_embed[adj][7] = (float) color / curr_graph.max_degree;
			}
		}
	}
	node_embed[node][6] = -0.01; // if node is colored, then its value is lowered for better selection in network
	node_embed[node][8] = 1; 	// node is marked as colored
}


/**
 * Number of adjacents of colored nodes and number of colored adjacents of colored nodes are
 * calculated and graph embedding for the selected graph is updated 
 * */
void update_adj_values()
{
	std::unordered_set<int> adj_of_colored;
	std::unordered_set<int> colored_adj_of_colored;
	auto curr_graph = graphs;
	auto graph_embed = graph_embeds;
	auto color_arr = color_arrs;


	if (color_arr == -1)
		continue;
	for (int u = curr_graph.row_ptr[v]; u < curr_graph.row_ptr[v + 1]; u++) // for each adjacent of colored node v
	{
		int &adj = curr_graph.col_ind[u];
		if (color_arr[adj] != -1)
		{
			colored_adj_of_colored.insert(adj);
		}
		adj_of_colored.insert(adj);
	}

	graph_embed[1] = adj_of_colored.size();
	graph_embed[2] = colored_adj_of_colored.size();
}


/**
 * Graph embedding for the selected graph is calculated again
 * */
void update_graph_embed()
{
	auto curr_graph = graphs;
	auto graph_embed = graph_embeds;
	auto color_arr = color_arrs;

	double degree_mean = (double)curr_graph.row_ptr.back() / curr_graph.num_nodes; // mean degree is calculated
	double uncolored_degree_sq_sum = 0;
	double colored_degree_sq_sum = 0;
	int colored_degree_sum = 0;
	int colored_size = 0;
	int uncolored_degree_sum = 0;
	int uncolored_size = 0;
	graph_embed[7] = 0;
	graph_embed[8] = 0;
	graph_embed[9] = 0;
	graph_embed[10] = 0;
	graph_embed[11] = 0; // mean degree of colored nodes
	graph_embed[12] = 0;
	graph_embed[13] = 0;
	graph_embed[14] = 0;
	graph_embed[3] = 0;
	graph_embed[5] = 0;

	for (int v = 0; v < curr_graph.num_nodes; v++)
	{
		int degree = curr_graph.row_ptr[v + 1] - curr_graph.row_ptr[v];
		if (color_arr[v] != -1)
		{
			if (degree > degree_mean)
			{
				graph_embed[7]++; // number of colored nodes with degree above mean
			}
			else
			{
				graph_embed[8]++; // number of colored nodes with degree below mean
			}
			colored_degree_sq_sum += degree * degree;
			colored_degree_sum += degree;
			colored_size++;
			graph_embed[3] += node_embeds[v][3]; // sum of closeness centrality of colored nodes
		}
		else if (color_arr[v] == -1)
		{
			if (degree > degree_mean)
			{
				graph_embed[9]++; // number of uncolored nodes with degree above mean
			}
			else
			{
				graph_embed[10]++; // number of uncolored nodes with degree above mean
			}
			uncolored_degree_sq_sum += degree * degree;
			uncolored_degree_sum += degree;
			uncolored_size++;
			graph_embed[5] += node_embeds[v][3]; // sum of closeness centrality of uncolored nodes
		}
	}
	graph_embed[4] = colored_size;
	graph_embed[6] = uncolored_size;
	graph_embed[11] = (double)colored_degree_sq_sum / colored_size;		// mean degree of colored
	graph_embed[12] = (double)uncolored_degree_sq_sum / uncolored_size;	// mean degree of uncolored
	graph_embed[13] = (double)colored_degree_sq_sum / curr_graph.num_nodes - (degree_mean * degree_mean); 	// variance of colored
	graph_embed[14] = (double)uncolored_degree_sq_sum / curr_graph.num_nodes - (degree_mean * degree_mean);	// variance of uncolored
	update_adj_values();	// updating adjacent colored node related embedding parameters
}


/**
 * For the whole batch, graph embeddings are updated and normalized with each other
 * */
extern "C" void update_graph_embeddings()
{
	update_graph_embed();
}


/**
 * For the graph given with the index, given node is colored and color is returned with
 * the color parameter
 * */
void color_graph(int node, int &color)
{
	auto curr_graph = graphs;
	auto color_arr = color_arrs;
	int n = node;

	std::vector<int> forbid_arr(curr_graph.num_nodes, -1);
	for (int edge = curr_graph.row_ptr[n]; edge < curr_graph.row_ptr[n + 1]; edge++)
	{	// for each adjacent of selected node
		int &adj = curr_graph.col_ind[edge];
		if (color_arr[adj] != -1) 	// if that adjacent is already colored
			forbid_arr[color_arr[adj]] = n;	// for the color of that adjacent, our selected node is forbidden
	}
	for (; color < curr_graph.num_nodes; color++)
	{
		if (forbid_arr[color] != n)
		{
			color_arr[n] = color; // first fitting color is selected
			break;
		}
	}
}


/**
 * For each graph in the batch, nodes to color are passed and the colors selected for these nodes
 * are calculated and returned. Size parameter is the size of colors array defied below.
 * Since coloring is applied for a whole batch, some graphs may be colored completely. For these graphs, 
 * nodes parameter is given as -1 and no color is set in this function
 * */
extern "C" int *color_batch(int *nodes, int *size)
{
	colors = 0;
	*size = 1;
	if(*(nodes) == -1) {	// if there is no node to color
			colors = -1;			// color is set to -1
			continue;
	}
	color_graph(*(nodes), colors); 	// else the graph is colored
	update_node_embed(*(nodes), colors);

	int res = 0;
	res = colors;

	return res;
}



