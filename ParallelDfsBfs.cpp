#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

void bfs(vector<vector<int>>& graph, int start, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    visited[start] = true;

    #pragma omp parallel
    {
        #pragma omp single
        {
            while (!q.empty()) {
                int vertex = q.front();
                q.pop();

                #pragma omp task firstprivate(vertex)
                {
                    for (int neighbor : graph[vertex]) {
                        if (!visited[neighbor]) {
                            q.push(neighbor);
                            visited[neighbor] = true;
                            #pragma omp task
                            bfs(graph, neighbor, visited);
                        }
                    }
                }
            }
        }
    }
}

void parallel_bfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    bfs(graph, start, visited);
}
void dfs(vector<vector<int>>& graph, int start, vector<bool>& visited) {
    stack<int> s;
    s.push(start);
    visited[start] = true;
#pragma omp parallel
    {
#pragma omp single
        {
            while (!s.empty()) {
                int vertex = s.top();
                s.pop();
#pragma omp task firstprivate(vertex)
                {
                    for (int neighbor : graph[vertex]) {
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                            visited[neighbor] = true;
#pragma omp task
                            dfs(graph, neighbor, visited);
                        }
                    }
                }
            }
        }
    }
}

void parallel_dfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    dfs(graph, start, visited);
}

int main() {
    // Create an undirected graph with 7 vertices
    vector<vector<int>> graph(7);
    graph[0] = {1, 2};
    graph[1] = {0, 2, 3, 4};
    graph[2] = {0, 1, 5, 6};
    graph[3] = {1, 4};
    graph[4] = {1, 3};
    graph[5] = {2};
    graph[6] = {2};

    // Run parallel BFS on the graph starting from vertex 0
    parallel_bfs(graph, 0);
    parallel_dfs(graph,0);

    return 0;
}

