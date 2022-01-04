#include <iostream>
#include <fstream>
using namespace std;

struct vertex{
   int index;  //index
   vertex *p;
};

struct edge{
   vertex *start;
   vertex *end;
};

class Graph{
public:
   vector<vector<edge>> adjL; //set to private
   vector<vector<edge>> adjL2;
   vector<vertex> head;
   int size;
   int e;
   void initialize(int vertice , int edges){
      head.resize(vertice);
      for(int i = 0 ; i < vertice ; i++){
         vertex j;
         j.index = i;
         head[i] = j;
      }

      adjL.resize(vertice);
      adjL2.resize(vertice);

      size = vertice;
      e = edges;
   }

   void buildComplete(int vertice){
      for(int i = 0 ; i < vertice - 1 ; i++){
         for(int j = i + 1 ; j < vertice ; j++) adjL[i].push_back({&head[i] , &head[j]});
      }
   }

   void buildNonEdge(int start , int ends){
      if(start < ends){
         for(int i = 0 ; i < adjL[start].size() ; i++){
            if(adjL[start][i].end->index == ends){
               adjL[start].erase(adjL[start].begin() + i);
               break;
            }
         }
      }
      else{
         for(int i = 0 ; i < adjL[ends].size() ; i++){
            if(adjL[ends][i].end->index == start){
               adjL[ends].erase(adjL[ends].begin() + i);
               break;
            }
         }
      }
      adjL2[start].push_back({&head[start] , &head[ends]});
      adjL2[ends].push_back({&head[ends] , &head[start]});
   }

   int numofV(){
      return size;
   }
   int numofE(){
      return e;
   }
};