; ModuleID = '/home/kranex/dev/rust-metaprogramming/examples/oxide/nbodies.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@llvm.used = appending global [2 x ptr] [ptr @gpu_integrate, ptr @gpu_n_bodies], section "llvm.metadata"

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @gpu_integrate(ptr captures(address_is_null) %v0, i64 %v1, ptr readonly captures(none) %v2, i64 %v3, float %v4) #0 {
entry:
  %v2.i4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5
  %v3.i5.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5
  %v4.i6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5
  %v5.i.i = zext nneg i32 %v2.i4.i to i64
  %v6.i7.i = zext nneg i32 %v3.i5.i to i64
  %v17.i.i = mul nuw nsw i64 %v5.i.i, %v6.i7.i
  %v7.i8.i = zext nneg i32 %v4.i6.i to i64
  %v18.i.i = add nuw nsw i64 %v17.i.i, %v7.i8.i
  %v4.i9.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5
  %v6.i10.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5
  %v13.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5
  %v15.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5
  %v14.i.i = icmp ne i32 %v13.i.i, 1
  %v16.i.i = icmp ne i32 %v15.i.i, 1
  %v17.i11.i.not9 = or i1 %v14.i.i, %v16.i.i
  %v5.i12.i = icmp ne i32 %v4.i9.i, 1
  %v7.i13.i = icmp ne i32 %v6.i10.i, 1
  %v8.not.not.i.i.not8 = or i1 %v5.i12.i, %v7.i13.i
  %.v18.i.i.not7 = or i1 %v8.not.not.i.i.not8, %v17.i11.i.not9
  %v19.not.i = icmp uge i64 %v18.i.i, %v1
  %v23.i = getelementptr inbounds nuw { [3 x float] }, ptr %v0, i64 %v18.i.i
  %.not = select i1 %.v18.i.i.not7, i1 true, i1 %v19.not.i
  %v42.i = select i1 %.v18.i.i.not7, i64 undef, i64 %v18.i.i
  %v20.not.not6 = icmp eq ptr %v0, null
  %v20.not.not = select i1 %.not, i1 true, i1 %v20.not.not6
  br i1 %v20.not.not, label %bb6, label %bb3

bb3:                                              ; preds = %entry
  %v34 = icmp ult i64 %v42.i, %v3
  br i1 %v34, label %bb4, label %bb8

bb4:                                              ; preds = %bb3
  %v36 = getelementptr inbounds { [3 x float] }, ptr %v2, i64 %v42.i
  %v38.unpack = load float, ptr %v36, align 4
  %v38.elt1 = getelementptr inbounds nuw i8, ptr %v36, i64 4
  %v38.unpack2 = load float, ptr %v38.elt1, align 4
  %v38.elt3 = getelementptr inbounds nuw i8, ptr %v36, i64 8
  %v38.unpack4 = load float, ptr %v38.elt3, align 4
  %v41 = load float, ptr %v23.i, align 4
  %v44 = fmul contract float %v4, %v38.unpack
  %v45 = fadd contract float %v44, %v41
  store float %v45, ptr %v23.i, align 4
  %v49 = getelementptr inbounds nuw i8, ptr %v23.i, i64 4
  %v50 = load float, ptr %v49, align 4
  %v53 = fmul contract float %v4, %v38.unpack2
  %v54 = fadd contract float %v53, %v50
  store float %v54, ptr %v49, align 4
  %v58 = getelementptr inbounds nuw i8, ptr %v23.i, i64 8
  %v59 = load float, ptr %v58, align 4
  %v62 = fmul contract float %v4, %v38.unpack4
  %v63 = fadd contract float %v62, %v59
  store float %v63, ptr %v58, align 4
  br label %bb6

bb6:                                              ; preds = %entry, %bb4
  ret void

bb8:                                              ; preds = %bb3
  tail call void @llvm.trap() #5
  unreachable
}

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @gpu_n_bodies(ptr readonly captures(none) %v0, i64 %v1, ptr captures(address_is_null) %v2, i64 %v3, float %v4, float %v5) #0 {
entry:
  %v2.i4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5
  %v3.i5.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5
  %v4.i6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5
  %v5.i.i = zext nneg i32 %v2.i4.i to i64
  %v6.i7.i = zext nneg i32 %v3.i5.i to i64
  %v17.i.i = mul nuw nsw i64 %v5.i.i, %v6.i7.i
  %v7.i8.i = zext nneg i32 %v4.i6.i to i64
  %v18.i.i = add nuw nsw i64 %v17.i.i, %v7.i8.i
  %v4.i9.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5
  %v6.i10.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5
  %v13.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5
  %v15.i.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5
  %v14.i.i = icmp ne i32 %v13.i.i, 1
  %v16.i.i = icmp ne i32 %v15.i.i, 1
  %v17.i11.i.not14 = or i1 %v14.i.i, %v16.i.i
  %v5.i12.i = icmp ne i32 %v4.i9.i, 1
  %v7.i13.i = icmp ne i32 %v6.i10.i, 1
  %v8.not.not.i.i.not13 = or i1 %v5.i12.i, %v7.i13.i
  %.v18.i.i.not12 = or i1 %v8.not.not.i.i.not13, %v17.i11.i.not14
  %v19.not.i = icmp uge i64 %v18.i.i, %v3
  %v23.i = getelementptr inbounds nuw { [3 x float] }, ptr %v2, i64 %v18.i.i
  %.not = select i1 %.v18.i.i.not12, i1 true, i1 %v19.not.i
  %v42.i = select i1 %.v18.i.i.not12, i64 undef, i64 %v18.i.i
  %v23.not.not11 = icmp eq ptr %v2, null
  %v23.not.not = select i1 %.not, i1 true, i1 %v23.not.not11
  br i1 %v23.not.not, label %bb11, label %bb3

bb3:                                              ; preds = %entry
  %v37 = icmp ult i64 %v42.i, %v1
  br i1 %v37, label %bb4, label %bb17

bb4:                                              ; preds = %bb3
  %v39 = getelementptr inbounds { [3 x float] }, ptr %v0, i64 %v42.i
  %v41.unpack = load float, ptr %v39, align 4
  %v41.elt1 = getelementptr inbounds nuw i8, ptr %v39, i64 4
  %v41.unpack2 = load float, ptr %v41.elt1, align 4
  %v41.elt3 = getelementptr inbounds nuw i8, ptr %v39, i64 8
  %v41.unpack4 = load float, ptr %v41.elt3, align 4
  %0 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #6
  %.not.i = icmp eq i32 %0, 0
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #6
  %.not1.i = icmp eq i32 %1, 0
  br label %bb15

bb8:                                              ; preds = %__nv_sqrtf.exit
  %v52 = load float, ptr %v23.i, align 4
  %v53 = fmul contract float %v4, %v118
  %v54 = fadd contract float %v53, %v52
  store float %v54, ptr %v23.i, align 4
  %v58 = getelementptr inbounds nuw i8, ptr %v23.i, i64 4
  %v59 = load float, ptr %v58, align 4
  %v60 = fmul contract float %v4, %v120
  %v61 = fadd contract float %v60, %v59
  store float %v61, ptr %v58, align 4
  %v65 = getelementptr inbounds nuw i8, ptr %v23.i, i64 8
  %v66 = load float, ptr %v65, align 4
  %v67 = fmul contract float %v4, %v122
  %v68 = fadd contract float %v67, %v66
  store float %v68, ptr %v65, align 4
  br label %bb11

bb11:                                             ; preds = %entry, %bb8
  ret void

bb15:                                             ; preds = %bb4, %__nv_sqrtf.exit
  %v9719 = phi i64 [ 1, %bb4 ], [ %v97, %__nv_sqrtf.exit ]
  %v4518 = phi i64 [ 0, %bb4 ], [ %v9719, %__nv_sqrtf.exit ]
  %v4417 = phi float [ 0.000000e+00, %bb4 ], [ %v122, %__nv_sqrtf.exit ]
  %v4316 = phi float [ 0.000000e+00, %bb4 ], [ %v120, %__nv_sqrtf.exit ]
  %v4215 = phi float [ 0.000000e+00, %bb4 ], [ %v118, %__nv_sqrtf.exit ]
  %v72 = getelementptr inbounds { [3 x float] }, ptr %v0, i64 %v4518
  %v74.unpack = load float, ptr %v72, align 4
  %v74.elt6 = getelementptr inbounds nuw i8, ptr %v72, i64 4
  %v74.unpack7 = load float, ptr %v74.elt6, align 4
  %v74.elt8 = getelementptr inbounds nuw i8, ptr %v72, i64 8
  %v74.unpack9 = load float, ptr %v74.elt8, align 4
  %v79 = fsub contract float %v74.unpack, %v41.unpack
  %v84 = fsub contract float %v74.unpack7, %v41.unpack2
  %v89 = fsub contract float %v74.unpack9, %v41.unpack4
  %v90 = fmul contract float %v79, %v79
  %v91 = fmul contract float %v84, %v84
  %v92 = fadd contract float %v90, %v91
  %v93 = fmul contract float %v89, %v89
  %v94 = fadd contract float %v92, %v93
  %v95 = fadd contract float %v5, %v94
  br i1 %.not.i, label %7, label %2

2:                                                ; preds = %bb15
  br i1 %.not1.i, label %5, label %3

3:                                                ; preds = %2
  %4 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v95) #6
  br label %__nv_sqrtf.exit

5:                                                ; preds = %2
  %6 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v95) #6
  br label %__nv_sqrtf.exit

7:                                                ; preds = %bb15
  br i1 %.not1.i, label %10, label %8

8:                                                ; preds = %7
  %9 = tail call float @llvm.nvvm.sqrt.rn.f(float %v95) #6
  br label %__nv_sqrtf.exit

10:                                               ; preds = %7
  %11 = tail call float @llvm.nvvm.sqrt.approx.f(float %v95) #6
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %3, %5, %8, %10
  %.0.i = phi float [ %4, %3 ], [ %6, %5 ], [ %9, %8 ], [ %11, %10 ]
  %v114 = fdiv contract float 1.000000e+00, %.0.i
  %v115 = fmul contract float %v114, %v114
  %v116 = fmul contract float %v114, %v115
  %v117 = fmul contract float %v79, %v116
  %v118 = fadd contract float %v4215, %v117
  %v119 = fmul contract float %v84, %v116
  %v120 = fadd contract float %v4316, %v119
  %v121 = fmul contract float %v89, %v116
  %v122 = fadd contract float %v4417, %v121
  %v97 = add i64 %v9719, 1
  %exitcond.not = icmp eq i64 %v9719, %v1
  br i1 %exitcond.not, label %bb8, label %bb15

bb17:                                             ; preds = %bb3
  tail call void @llvm.trap() #5
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

; Function Attrs: nofree nosync nounwind memory(none)
declare noundef i32 @__nvvm_reflect(ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.ftz.f(float) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.rn.f(float) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.sqrt.approx.f(float) #4

attributes #0 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: write) }
attributes #1 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nosync nounwind memory(none) "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent }
attributes #6 = { nounwind }

!llvm.ident = !{!0}
!nvvmir.version = !{!1}

!0 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!1 = !{i32 2, i32 0}
