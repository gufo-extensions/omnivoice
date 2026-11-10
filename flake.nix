{
  description = "Gufo with the OmniVoice extension";
  inputs.gufo.url = "github:gufo-org/gufo/0e2de43ec01540079fbec30492b06704796d725e";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
