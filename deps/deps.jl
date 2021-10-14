using Pkg

exatron = Pkg.PackageSpec(url="https://github.com/exanauts/ExaTron.jl.git", rev="youngdae/multiperiod")
Pkg.add([exatron])