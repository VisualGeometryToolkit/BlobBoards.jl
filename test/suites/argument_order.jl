# =============================================================================
# The retired argument orders bind no method.
#
# The stack's argument-order rule (VisualGeometryToolkitCore's CLAUDE.md: the model or
# shape first, then the data it acts on, then parameters) moved the given
# camera or homography, the required tools (gallery, network) and the shapes
# ahead of the images and detections. Many of those arguments were untyped, so
# a call in the OLD order would have bound — every value in the wrong slot —
# and failed somewhere downstream, or not at all. The signatures are typed so
# such a call is a MethodError instead; this pins it. `applicable` and
# `hasmethod` read the method table only: nothing here runs a pipeline.
#
# The second round (2026-09-17) applied Julia's style-guide order — function,
# IO/RNG, mutated inputs, type, the rest — so a `!` function leads with what it
# mutates, a function that only writes files takes no `!`, a keyword the caller
# read back is returned or called back instead, and a type argument is
# positional. Where every argument is a string (the training entry points) a
# file check raises the ArgumentError; those checks run before any work, so
# they are pinned here too.
# (The camera-bearing entry points on real fixtures — `find_board`,
# `board_quad`, the Makie viewers — are checked beside their own tests.)
# =============================================================================

using StaticArrays: SMatrix
using VisualGeometryToolkitCore: CameraModel, LogicalIntrinsics, CameraCalibrationMat, Pinhole, EllipticFrame, Point2, px

@testset "retired argument orders bind no method" begin
    img = zeros(Float32, 16, 16)
    gallery = Dict{String, Any}()
    net = "descriptor.onnx"                      # a path is a valid network argument
    cam = CameraModel(LogicalIntrinsics(CameraCalibrationMat(900.0 * px, [8.0px, 8.0px])), Pinhole())
    frames = [EllipticFrame(Point2(8.0, 8.0), 2.0)]
    discs = [BlobBoards.Disc(8.0, 8.0, 2.0)]
    H = SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, 1)

    @testset "task verbs: the gallery and the network lead, the images follow" begin
        @test !applicable(find_boards, img, gallery, net)
        @test !applicable(find_boards, [img], gallery, net)
        @test !applicable(track_boards, [img], gallery, net)
        @test !applicable(track_boards, "clip.mp4", gallery, net)
        @test !applicable(BlobBoards.annotate_blobs, img, gallery, net)
        @test !applicable(BlobBoards.annotate_blobs, "clip.mp4", gallery, net)
        @test !applicable(BlobBoards.build_annotations, ["a.jpg"], gallery, net)
        @test !applicable(
            BlobBoards.annotate_frame!, "out", "out/a.jld2", "a.jpg", img, gallery, net,
            Dict{String, String}(), Set{String}(),
        )
        # round 1's destinations-first order: the output directory is written,
        # not mutated, so the mutated `Set` leads
        @test !applicable(
            BlobBoards.annotate_frame!, "out", Set{String}(), gallery, net,
            Dict{String, String}(), "out/a.jld2", "a.jpg", img,
        )
        @test applicable(
            BlobBoards.annotate_frame!, Set{String}(), "out", gallery, net,
            Dict{String, String}(), "out/a.jld2", "a.jpg", img,
        )
        @test !applicable(BlobBoards.prepare_image_features, img, net)
        @test !applicable(extract_image_features, img, net)
        @test !applicable(BlobBoards.trace_pipeline, img, nothing)
        @test !applicable(BlobBoards.trace_pipeline, img, cam)
        @test !hasmethod(extract_board_features, Tuple{NamedTuple, BlobBoards.AbstractDescriptorNetwork})
        @test !hasmethod(BlobBoards.build_absolute_tracks, Tuple{String, Dict, BlobBoards.DescriptorNetwork})
        # …and the new orders bind
        @test applicable(find_boards, gallery, net, img)
        @test applicable(find_boards, gallery, net, [img])
        @test applicable(BlobBoards.prepare_image_features, net, img)
        @test applicable(BlobBoards.trace_pipeline, nothing, img)
        @test hasmethod(extract_board_features, Tuple{BlobBoards.AbstractDescriptorNetwork, NamedTuple})
        @test hasmethod(BlobBoards.build_absolute_tracks, Tuple{Dict, BlobBoards.DescriptorNetwork, String})
    end

    @testset "shapes lead the image they are sampled from" begin
        @test !applicable(BlobBoards.canonicalize, img, frames)
        @test !applicable(BlobBoards.adapt, img, frames)
        @test !applicable(BlobBoards.adapt, img, discs)
        @test applicable(BlobBoards.canonicalize, frames, img)
        @test applicable(BlobBoards.adapt, frames, img)
        @test applicable(BlobBoards.adapt, discs, img)
    end

    @testset "a given homography leads the frames it relates" begin
        @test !applicable(BlobBoards.symmetric_iou_match, frames, frames, H)
        @test applicable(BlobBoards.symmetric_iou_match, H, frames, frames)
    end

    @testset "the model leads, the data it is applied to follows" begin
        @test !applicable(BlobBoards.describe_detections, img, frames, net)
        @test applicable(BlobBoards.describe_detections, net, frames, img)
        @test !applicable(BlobBoards.describe_frames, img, frames, net)
        @test applicable(BlobBoards.describe_frames, net, frames, img)
        @test !applicable(BlobBoards.adapt_detections, img, frames)
        @test applicable(BlobBoards.adapt_detections, frames, img)
        @test !applicable(BlobBoards._canon_describe, nothing, net, img, frames)
        @test applicable(BlobBoards._canon_describe, nothing, net, frames, img)
        # all-string signatures: the model-file check names the order
        dir = mktempdir()
        @test_throws ArgumentError BlobBoards.load_board_gallery(dir, joinpath(dir, "descriptor.onnx"))
        @test_throws ArgumentError BlobBoards.build_tracks_for_sequences(dir, dir, joinpath(dir, "descriptor.onnx"))
    end

    @testset "a `!` function leads with what it mutates" begin
        @test !applicable(BlobBoards._run_matching_phase!, frames, gallery, Tuple[])
        @test applicable(BlobBoards._run_matching_phase!, gallery, Tuple[], frames)
    end

    @testset "file writers take no `!`; a tracks writer is not a `save` method" begin
        @test !isdefined(BlobBoards, :append_log!)
        @test !isdefined(BlobBoards, :check_annotation_hashes!)
        @test isdefined(BlobBoards, :append_log) && isdefined(BlobBoards, :check_annotation_hashes)
        @test !hasmethod(BlobBoards.save, Tuple{String, Dict})
        # public, not exported: Makie exports a `save` too, and `using BlobBoards, CairoMakie` is the authoring setup
        @test Base.ispublic(BlobBoards, :save) && !Base.isexported(BlobBoards, :save)
        @test hasmethod(BlobBoards.save_tracks, Tuple{String, Dict})
    end

    @testset "outputs are returned or called back, never keywords filled in" begin
        video = Matrix{Float32}[]
        @test !hasmethod(track_boards, Tuple{Dict{String, Any}, String, typeof(video)}, (:trace,))
        @test hasmethod(track_boards, Tuple{Dict{String, Any}, String, typeof(video)}, (:on_frame,))
        @test !hasmethod(BlobBoards.find_board, Tuple{Nothing, Vector{Any}, Vector{Any}}, (:deps_out,))
        @test hasmethod(BlobBoards.find_board, Tuple{Nothing, Vector{Any}, Vector{Any}}, (:on_guided_read,))
        tracks_sig = Tuple{Dict, BlobBoards.DescriptorNetwork, String}
        @test !hasmethod(BlobBoards.build_absolute_tracks, tracks_sig, (:view_stats,))
        @test hasmethod(BlobBoards.build_absolute_tracks, tracks_sig, (:on_view_stats,))
    end

    @testset "a type argument is positional" begin
        # rejected before any placement runs
        @test_throws ArgumentError blob_pattern(64, 64; pixel_type = Float32)
        @test_throws ArgumentError blob_pattern(Size2(width = 64, height = 64); pixel_type = Float32)
        @test_throws ArgumentError blob_pattern(Float32, Size2(width = 64, height = 64); pixel_type = Float32)
    end
end
