#include <TH.h>
#include <luaT.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)

#include "generic/hungarian.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libhungarian(lua_State *L)
{
  torch_FloatHungarian_init(L);
  torch_DoubleHungarian_init(L);

  return 1;
}
