{
  description = "Gufo with the OmniVoice extension";
  inputs.gufo.url = "github:gufo-org/gufo/5f9612c3b19ab6676b842095bc88bb9b32da5cf0";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
