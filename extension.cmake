# OmniVoice (k2-fsa) text-to-speech extension.

# Host side: configuration, probe and the CPU-only registration.
gufo_extension_target(
  NAME gufo_omnivoice
  SOURCES
    config.cpp
    safetensors.cpp
    unicode.cpp
    tokenizer.cpp
    text.cpp
    prompt.cpp
    audio.cpp
    decoding.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    ICU::uc
)
if(TARGET gufo_omnivoice)
  target_include_directories(gufo_omnivoice PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_omnivoice PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# Device side: the generator's kernels and runtime.
gufo_extension_target(
  NAME gufo_omnivoice_hip
  SOURCES
    gemm.cpp
    generator.cpp
    codec.cpp
    codec_encoder.cpp
    synthesizer.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
    roc::hipblas
    roc::hipblaslt
)
if(TARGET gufo_omnivoice_hip)
  gufo_extension_hip_sources(gufo_omnivoice_hip
    SOURCES
      kernels/generator_kernels.hip
      kernels/codec_kernels.hip
      kernels/encoder_kernels.hip
    OPTIONS
      -O3 -Wno-unused-command-line-argument
  )
  target_include_directories(gufo_omnivoice_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# Speech synthesis provider: the TtsRunner and its registration.
gufo_extension_target(
  NAME gufo_omnivoice_tts
  SOURCES
    runner.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_omnivoice_hip
    gufo_omnivoice
    gufo_ext_runtime
    gufo_ext_hip
)

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_gufo_omnivoice_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_omnivoice_test_libs gufo_ext_hip)
endif()

gufo_extension_target(
  NAME omnivoice_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_config_test)
  add_test(NAME omnivoice.config COMMAND omnivoice_config_test)
  set_tests_properties(omnivoice.config PROPERTIES LABELS "omnivoice;cpu")
endif()

gufo_extension_target(
  NAME omnivoice_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_probe_test)
  add_test(NAME omnivoice.probe COMMAND omnivoice_probe_test samples)
  set_tests_properties(omnivoice.probe PROPERTIES LABELS "omnivoice;cpu")
  add_test(NAME omnivoice.probe_model COMMAND omnivoice_probe_test model)
  set_tests_properties(omnivoice.probe_model PROPERTIES
    LABELS "omnivoice;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME omnivoice_safetensors_test
  SOURCES
    tests/safetensors_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_safetensors_test)
  add_test(NAME omnivoice.safetensors COMMAND omnivoice_safetensors_test samples)
  set_tests_properties(omnivoice.safetensors PROPERTIES LABELS "omnivoice;cpu")
  add_test(NAME omnivoice.safetensors_model
    COMMAND omnivoice_safetensors_test model)
  set_tests_properties(omnivoice.safetensors_model PROPERTIES
    LABELS "omnivoice;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

set(_gufo_omnivoice_fixtures "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures")
gufo_extension_target(
  NAME omnivoice_tokenizer_test
  SOURCES
    tests/tokenizer_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_tokenizer_test)
  target_compile_definitions(omnivoice_tokenizer_test PRIVATE
    GUFO_OMNIVOICE_TOKENIZER_GOLDEN="${_gufo_omnivoice_fixtures}/tokenizer_golden.json")
  add_test(NAME omnivoice.tokenizer COMMAND omnivoice_tokenizer_test)
  set_tests_properties(omnivoice.tokenizer PROPERTIES
    LABELS "omnivoice;cpu;tokenizer;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME omnivoice_text_test
  SOURCES
    tests/text_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_text_test)
  target_compile_definitions(omnivoice_text_test PRIVATE
    GUFO_OMNIVOICE_PROMPT_GOLDEN="${_gufo_omnivoice_fixtures}/prompt_golden.json")
  add_test(NAME omnivoice.text COMMAND omnivoice_text_test text)
  set_tests_properties(omnivoice.text PROPERTIES LABELS "omnivoice;cpu")
  add_test(NAME omnivoice.prompt COMMAND omnivoice_text_test prompt)
  set_tests_properties(omnivoice.prompt PROPERTIES
    LABELS "omnivoice;cpu;tokenizer;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME omnivoice_audio_test
  SOURCES
    tests/audio_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_audio_test)
  target_compile_definitions(omnivoice_audio_test PRIVATE
    GUFO_OMNIVOICE_AUDIO_GOLDEN="${_gufo_omnivoice_fixtures}/audio_golden.json")
  add_test(NAME omnivoice.audio COMMAND omnivoice_audio_test)
  set_tests_properties(omnivoice.audio PROPERTIES LABELS "omnivoice;cpu")
endif()

set(_gufo_omnivoice_generator_golden
  "${_gufo_omnivoice_fixtures}/generator_golden.json")
gufo_extension_target(
  NAME omnivoice_decoding_test
  SOURCES
    tests/decoding_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_decoding_test)
  target_compile_definitions(omnivoice_decoding_test PRIVATE
    GUFO_OMNIVOICE_GENERATOR_GOLDEN="${_gufo_omnivoice_generator_golden}")
  add_test(NAME omnivoice.decoding COMMAND omnivoice_decoding_test)
  set_tests_properties(omnivoice.decoding PROPERTIES LABELS "omnivoice;cpu")
endif()

gufo_extension_target(
  NAME omnivoice_generator_test
  SOURCES
    tests/generator_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice_hip
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_generator_test)
  target_compile_definitions(omnivoice_generator_test PRIVATE
    GUFO_OMNIVOICE_GENERATOR_GOLDEN="${_gufo_omnivoice_generator_golden}")
  add_test(NAME omnivoice.generator COMMAND omnivoice_generator_test)
  set_tests_properties(omnivoice.generator PROPERTIES
    LABELS "omnivoice;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME omnivoice_codec_test
  SOURCES
    tests/codec_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice_hip
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_codec_test)
  target_compile_definitions(omnivoice_codec_test PRIVATE
    GUFO_OMNIVOICE_GENERATOR_GOLDEN="${_gufo_omnivoice_generator_golden}")
  add_test(NAME omnivoice.codec COMMAND omnivoice_codec_test)
  set_tests_properties(omnivoice.codec PROPERTIES
    LABELS "omnivoice;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME omnivoice_synthesize
  SOURCES
    tools/synthesize/synthesize.cpp
  LINK_LIBRARIES
    gufo_omnivoice_hip
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)

gufo_extension_target(
  NAME omnivoice_synthesis_test
  SOURCES
    tests/synthesis_test.cpp
  LINK_LIBRARIES
    gufo_omnivoice_hip
    gufo_omnivoice
    gufo_ext_runtime
    ${_gufo_omnivoice_test_libs}
)
if(TARGET omnivoice_synthesis_test)
  add_test(NAME omnivoice.synthesis COMMAND omnivoice_synthesis_test)
  set_tests_properties(omnivoice.synthesis PROPERTIES
    LABELS "omnivoice;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()
