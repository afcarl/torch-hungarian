#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/hungarian.c"
#else

#include "hungarian.h"

int torch_(Main_Hungarian)(lua_State *L) {
   THTensor *src = luaT_checkudata(L, 1, torch_Tensor);
   THTensor *dst = luaT_checkudata(L, 2, torch_Tensor);
   THLongTensor *ind = luaT_checkudata(L, 3, "torch.LongTensor");

   const int batches = src->size[0];
   const int height = src->size[1];
   const int width = src->size[2];
   int b;
   #pragma omp parallel for private(b)
   for (b = 0; b < batches; ++b) {
      int x, y;
      int tmp_array[height * width];
      for (y = 0; y < height; ++y) {
         for (x = 0; x < width; ++x) {
            tmp_array[y*width+x] = THTensor_(get3d)(src, b, y, x)*1000.0;
         }
      }

      int** tmp_matrix = array_to_matrix(tmp_array,height,width);
      hungarian_problem_t p;
      int matrix_size = hungarian_init(&p,tmp_matrix,height,width,HUNGARIAN_MODE_MINIMIZE_COST);

      for (y = 0; y < height; ++y) {
         for (x = 0; x < width; ++x) {
            THTensor_(set3d)(dst, b, y, x, p.cost[y][x]/1000.0);
         }
      }

      hungarian_solve(&p);

      for (y = 0; y < height; ++y) {
         for (x = 0; x < width; ++x) {
            THLongTensor_set3d(ind, b, y, x, (long)p.assignment[y][x]);
         }
      }

      hungarian_free(&p);

      free(tmp_matrix);
   }
      
   return 1;
}

static const struct luaL_Reg torch_(Hungarian__) [] = {
  {"hungarian", torch_(Main_Hungarian)},
  {NULL, NULL}
};

static void torch_(Hungarian_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, torch_(Hungarian__), "torch");
  lua_pop(L,1);
}

#endif
