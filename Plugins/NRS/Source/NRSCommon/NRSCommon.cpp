#include "NRSCommon.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FNRSCommonModule, NRSCommon)

void FNRSCommonModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSCommon module has started."));
}

void FNRSCommonModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("NRSCommon module is shutting down."));
}
