using Pkg

exatron = Pkg.PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="ms/ka")
Pkg.add([exatron])