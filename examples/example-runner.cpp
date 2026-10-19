#include "base/example.h"

rhi::ExampleDesc getExampleSurface();
rhi::ExampleDesc getExampleTriangle();
rhi::ExampleDesc getExampleShaderToy();
rhi::ExampleDesc getExampleRaster();
rhi::ExampleDesc getExampleParticles();
rhi::ExampleDesc getExamplePathTracer();

int main(int argc, const char** argv)
{
    const rhi::ExampleDesc examples[] = {
        getExampleSurface(),
        getExampleTriangle(),
        getExampleShaderToy(),
        getExampleRaster(),
        getExampleParticles(),
        getExamplePathTracer(),
    };
    return rhi::detail::runExamples(argc, argv, examples);
}
