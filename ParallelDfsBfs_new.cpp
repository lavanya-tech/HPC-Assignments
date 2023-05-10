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
void nbfs(vector<vector<int>>& graph, int start, vector<bool>& visited) {
    queue<int> q;
    q.push(start);
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        visited[vertex]=true;
        for (int neighbor : graph[vertex]) {
            if (!visited[neighbor]) {
                q.push(neighbor);
            }
        }
    }
}

void normal_bfs(vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    nbfs(graph, start, visited);
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
    // cout<<"Enter number of vertices"<<endl;
    // int n;
    // cin>>n;
    vector<vector<int>> graph(7);
    // int edges;
    // cout<<"Enter number of edges"<<endl;
    // cin>>edges;
    // cout<<"enter the edges"<<endl;
    // for(int i = 0;i < edges;i++)
    // {
    //     cin>>u;
    //     cin>>v;
    //     graph[u].push_back(v);
    //     graph[v].push_back(u);
    //     cout<<endl;
    // }
    
    double start_time, end_time;
    double start_time1, end_time1;
   
    graph[0] = {1, 2};
    graph[1] = {0, 2, 3, 4};
    graph[2] = {0, 1, 5, 6};
    graph[3] = {1, 4};
    graph[4] = {1, 3};
    graph[5] = {2};
    graph[6] = {2};
    start_time = omp_get_wtime();
    parallel_bfs(graph, 0);
    end_time = omp_get_wtime();
    cout<<"Parallel bfs took"<<end_time - start_time<<endl;
    start_time1 = omp_get_wtime();
    normal_bfs(graph, 0);
    end_time1 = omp_get_wtime();
    cout<<"normal bfs took "<<end_time1 - start_time<<endl;
    start_time1 = omp_get_wtime();

    cout<<"dfs output"<<endl;
    parallel_dfs(graph,0);
    end_time1 = omp_get_wtime();
    cout<<"parallel dfs took "<<end_time1 - start_time1<<endl;

    

    return 0;
}

