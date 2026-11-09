#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "probe.hpp"

// Without HIP the provider's library isn't built; registering its probe
// still lets Gufo name the model it can't run here.
void omnivoice_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  registry.AddTtsProvider({
      .id = "omnivoice",
      .probe = &gufo::models::omnivoice::ProbeOmniVoice,
      .resolve = &gufo::models::omnivoice::ResolveOmniVoiceArtifacts,
      .create = nullptr,
  });
#else
  (void)registry;
#endif
}
