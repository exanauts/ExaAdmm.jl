using Pkg

Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "..")))
# when first running instantiate
Pkg.instantiate()

using Documenter
using ExaAdmm

makedocs(
    sitename = "ExaAdmm.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX()
    ),
    modules = [ExaAdmm],
    repo = "https://github.com/exanauts/ExaAdmm.jl/blob/{commit}{path}#{line}",
    strict = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "How to Implement New Model" => "dev.md",
    ]
)

deploydocs(
    repo = "github.com/exanauts/ExaAdmm.jl.git",
    target = "build",
    devbranch = "main",
    devurl = "main",
    push_preview = true,
)
