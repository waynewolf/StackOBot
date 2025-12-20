#include "NRSSuperFrame.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FNRSSuperFrameModule, NRSSuperFrame)

void FNRSSuperFrameModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSSuperFrame module has started."));
}

void FNRSSuperFrameModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSSuperFrame module is shutting down."));
}
