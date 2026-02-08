// classic API header files
#include "c4d_plugin.h"
#include "c4d_resource.h"

namespace atmosphere {
Bool RegisterShader();
}


::Bool PluginStart()
{
  atmosphere::RegisterShader();

  return true;
}

void PluginEnd()
{
  // free resources
}

::Bool PluginMessage(::Int32 id, void* data)
{
  switch (id)
  {
    case C4DPL_INIT_SYS:
    {
      // load resources defined in the the optional "res" folder
      if (!g_resource.Init())
        return false;

      return true;
    }
  }

  return true;
}
