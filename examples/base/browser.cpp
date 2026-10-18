#include <emscripten.h>

namespace rhi::detail {

// Suspend the existing runner until the browser is ready for another frame.
EM_ASYNC_JS(void, waitForBrowserFrame, (), { await new Promise(requestAnimationFrame); });

} // namespace rhi::detail
