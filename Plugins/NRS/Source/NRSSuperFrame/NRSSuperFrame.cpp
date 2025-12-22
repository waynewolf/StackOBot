#include "NRSSuperFrame.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(NRSSuperFrameModule, NRSSuperFrame)

void NRSSuperFrameModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSSuperFrame module has started."));
}

void NRSSuperFrameModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSSuperFrame module is shutting down."));
}
