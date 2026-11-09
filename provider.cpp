#include <gufo/ext/generated/omnivoice_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "probe.hpp"
#include "runner.hpp"

void omnivoice_provider_register(gufo::ext::Registry& registry) {
  namespace omnivoice = gufo::models::omnivoice;
  registry.AddProvider(
      gufo::ext::generated::extension_omnivoice::kProvider_omnivoice);
  registry.AddTtsProvider({
      .id = "omnivoice",
      .probe = &omnivoice::ProbeOmniVoice,
      .resolve = &omnivoice::ResolveOmniVoiceArtifacts,
      .create = nullptr,
      .create_with_options = &omnivoice::CreateOmniVoiceRunner,
  });
}
