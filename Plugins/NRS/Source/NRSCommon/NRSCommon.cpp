#include "NRSCommon.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(NRSCommonModule, NRSCommon)

void NRSCommonModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSCommon module has started."));
}

void NRSCommonModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSCommon module is shutting down."));
}
