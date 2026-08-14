; ModuleID = '/home/kranex/dev/rust-metaprogramming/examples/oxide/raytracer.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@llvm.used = appending global [1 x ptr] [ptr @raytracing], section "llvm.metadata"

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @raytracing(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(address_is_null) %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v6.i = zext i32 %v4 to i64
  %v7.i = icmp eq i32 %v4, 0
  br i1 %v7.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb24.i

bb3.i:                                            ; preds = %bb24.i
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #5
  %v17.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5
  %v18.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #5
  %narrow.i = mul nuw nsw i32 %v13.i, %v17.i
  %narrow3.i = add nuw nsw i32 %narrow.i, %v18.i
  %v68.i = zext nneg i32 %narrow3.i to i64
  %v70.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5
  %v23.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5
  %v24.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5
  %v25.i = zext nneg i32 %v70.i to i64
  %v26.i = zext nneg i32 %v23.i to i64
  %v27.i = zext nneg i32 %v24.i to i64
  %v75.i = mul nuw nsw i64 %v25.i, %v26.i
  %v76.i = add nuw nsw i64 %v75.i, %v27.i
  %v30.not.i.not = icmp samesign ult i64 %v76.i, %v6.i
  %v42.i = shl nuw nsw i64 %v68.i, 32
  %v43.i = or i64 %v42.i, %v76.i
  %spec.select17 = select i1 %v30.not.i.not, i64 %v43.i, i64 undef
  br label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit

bb24.i:                                           ; preds = %entry
  %v57.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5
  %v58.i = icmp ne i32 %v57.i, 1
  %v59.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5
  %v60.i = icmp ne i32 %v59.i, 1
  %v61.i = or i1 %v58.i, %v60.i
  br i1 %v61.i, label %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, label %bb3.i

_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit: ; preds = %bb3.i, %entry, %bb24.i
  %v51.i = phi i1 [ false, %entry ], [ false, %bb24.i ], [ %v30.not.i.not, %bb3.i ]
  %v52.i = phi i64 [ undef, %entry ], [ undef, %bb24.i ], [ %spec.select17, %bb3.i ]
  br i1 %v51.i, label %bb3, label %bb27

bb3:                                              ; preds = %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit
  %v25 = uitofp i32 %v4 to float
  %v27 = trunc i64 %v52.i to i32
  %v30 = lshr i64 %v52.i, 32
  %v31 = trunc nuw i64 %v30 to i32
  %v32 = uitofp i32 %v27 to float
  %v33 = fmul contract float %v25, 5.000000e-01
  %v34 = fsub contract float %v32, %v33
  %v35 = uitofp i32 %v31 to float
  %v36 = fsub contract float %v35, %v33
  br label %bb31

bb7:                                              ; preds = %bb20
  %v10.i = and i64 %v52.i, 4294967295
  %v14.not.i = icmp samesign uge i64 %v10.i, %v6.i
  %v22.i = mul nuw nsw i64 %v30, %v6.i
  %v23.i13 = add nuw nsw i64 %v22.i, %v10.i
  %v26.not.i = icmp uge i64 %v23.i13, %v3
  %.not = select i1 %v14.not.i, i1 true, i1 %v26.not.i
  %v106.not19 = icmp eq ptr %v2, null
  %v106.not = select i1 %.not, i1 true, i1 %v106.not19
  br i1 %v106.not, label %bb27, label %bb23

bb8:                                              ; preds = %bb31
  %v51 = getelementptr inbounds nuw float, ptr %v0, i64 %v46
  %v52 = load float, ptr %v51, align 4
  %v53 = add nuw nsw i64 %v45, 4
  %v54 = icmp ult i64 %v53, %v1
  br i1 %v54, label %bb9, label %bb35

bb9:                                              ; preds = %bb8
  %v59 = add nuw nsw i64 %v45, 5
  %v60 = icmp ult i64 %v59, %v1
  br i1 %v60, label %bb10, label %bb36

bb10:                                             ; preds = %bb9
  %v56 = getelementptr inbounds nuw float, ptr %v0, i64 %v53
  %v57 = load float, ptr %v56, align 4
  %v58 = fsub contract float %v34, %v57
  %v62 = getelementptr inbounds nuw float, ptr %v0, i64 %v59
  %v63 = load float, ptr %v62, align 4
  %v64 = fsub contract float %v36, %v63
  %v65 = fmul contract float %v58, %v58
  %v66 = fmul contract float %v64, %v64
  %v67 = fadd contract float %v65, %v66
  %v68 = fmul contract float %v52, %v52
  %v69 = fcmp uge float %v67, %v68
  br i1 %v69, label %bb14, label %bb11

bb11:                                             ; preds = %bb10
  %v71 = fsub contract float %v68, %v65
  %v72 = fsub contract float %v71, %v66
  %0 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #6
  %.not.i = icmp eq i32 %0, 0
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #6
  %.not1.i = icmp eq i32 %1, 0
  br i1 %.not.i, label %7, label %2

2:                                                ; preds = %bb11
  br i1 %.not1.i, label %5, label %3

3:                                                ; preds = %2
  %4 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v72) #6
  br label %__nv_sqrtf.exit

5:                                                ; preds = %2
  %6 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v72) #6
  br label %__nv_sqrtf.exit

7:                                                ; preds = %bb11
  br i1 %.not1.i, label %10, label %8

8:                                                ; preds = %7
  %9 = tail call float @llvm.nvvm.sqrt.rn.f(float %v72) #6
  br label %__nv_sqrtf.exit

10:                                               ; preds = %7
  %11 = tail call float @llvm.nvvm.sqrt.approx.f(float %v72) #6
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %3, %5, %8, %10
  %.0.i = phi float [ %4, %3 ], [ %6, %5 ], [ %9, %8 ], [ %11, %10 ]
  %12 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #6
  %.not1.i10 = icmp eq i32 %12, 0
  br i1 %.not.i, label %18, label %13

13:                                               ; preds = %__nv_sqrtf.exit
  br i1 %.not1.i10, label %16, label %14

14:                                               ; preds = %13
  %15 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v68) #6
  br label %__nv_sqrtf.exit11

16:                                               ; preds = %13
  %17 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v68) #6
  br label %__nv_sqrtf.exit11

18:                                               ; preds = %__nv_sqrtf.exit
  br i1 %.not1.i10, label %21, label %19

19:                                               ; preds = %18
  %20 = tail call float @llvm.nvvm.sqrt.rn.f(float %v68) #6
  br label %__nv_sqrtf.exit11

21:                                               ; preds = %18
  %22 = tail call float @llvm.nvvm.sqrt.approx.f(float %v68) #6
  br label %__nv_sqrtf.exit11

__nv_sqrtf.exit11:                                ; preds = %14, %16, %19, %21
  %.0.i9 = phi float [ %15, %14 ], [ %17, %16 ], [ %20, %19 ], [ %22, %21 ]
  %v140 = add nuw nsw i64 %v45, 6
  %v141 = icmp ult i64 %v140, %v1
  br i1 %v141, label %bb12, label %bb40

bb12:                                             ; preds = %__nv_sqrtf.exit11
  %v139 = fdiv contract float %.0.i, %.0.i9
  %v75 = getelementptr inbounds nuw float, ptr %v0, i64 %v140
  %v76 = load float, ptr %v75, align 4
  %v77 = fadd contract float %.0.i, %v76
  br label %bb14

bb14:                                             ; preds = %bb10, %bb12
  %v78 = phi float [ %v139, %bb12 ], [ 0.000000e+00, %bb10 ]
  %v79 = phi float [ %v77, %bb12 ], [ -9.999900e+04, %bb10 ]
  %v80 = fcmp ule float %v79, %v4044
  br i1 %v80, label %bb20, label %bb15

bb15:                                             ; preds = %bb14
  %v84 = getelementptr inbounds nuw float, ptr %v0, i64 %v45
  %v85 = load float, ptr %v84, align 4
  %v86 = fmul contract float %v78, %v85
  %v90 = getelementptr inbounds nuw i8, ptr %v84, i64 4
  %v91 = load float, ptr %v90, align 4
  %v92 = fmul contract float %v78, %v91
  %v96 = getelementptr inbounds nuw i8, ptr %v84, i64 8
  %v97 = load float, ptr %v96, align 4
  %v98 = fmul contract float %v78, %v97
  br label %bb20

bb20:                                             ; preds = %bb14, %bb15
  %v99 = phi float [ %v86, %bb15 ], [ %v3741, %bb14 ]
  %v100 = phi float [ %v92, %bb15 ], [ %v3842, %bb14 ]
  %v101 = phi float [ %v98, %bb15 ], [ %v3943, %bb14 ]
  %v102 = phi float [ %v79, %bb15 ], [ %v4044, %bb14 ]
  %v121 = add nuw nsw i64 %v12146, 1
  %exitcond = icmp eq i64 %v121, 21
  br i1 %exitcond, label %bb7, label %bb31

bb23:                                             ; preds = %bb7
  %v30.i = getelementptr inbounds nuw { [4 x float] }, ptr %v2, i64 %v23.i13
  %v103.repack5 = getelementptr inbounds nuw i8, ptr %v30.i, i64 12
  %v103.repack3 = getelementptr inbounds nuw i8, ptr %v30.i, i64 8
  %v103.repack1 = getelementptr inbounds nuw i8, ptr %v30.i, i64 4
  %v114 = fmul contract float %v101, 2.550000e+02
  %v113 = fmul contract float %v100, 2.550000e+02
  %v112 = fmul contract float %v99, 2.550000e+02
  store float %v112, ptr %v30.i, align 16
  store float %v113, ptr %v103.repack1, align 4
  store float %v114, ptr %v103.repack3, align 8
  store float 2.550000e+02, ptr %v103.repack5, align 4
  br label %bb27

bb27:                                             ; preds = %bb7, %_RINvNtNtCsiWbdGYq5HZh_11cuda_device6thread10___internal16index_2d_runtimeNtCs6HixiQUBNAA_9raytracer4RgbaNtB2_13UnknownDomainNtB2_17NativeCoordinatesEB1b_.exit, %bb23
  ret void

bb31:                                             ; preds = %bb3, %bb20
  %v12146 = phi i64 [ 1, %bb3 ], [ %v121, %bb20 ]
  %v4145 = phi i64 [ 0, %bb3 ], [ %v12146, %bb20 ]
  %v4044 = phi float [ -9.999900e+04, %bb3 ], [ %v102, %bb20 ]
  %v3943 = phi float [ 0.000000e+00, %bb3 ], [ %v101, %bb20 ]
  %v3842 = phi float [ 0.000000e+00, %bb3 ], [ %v100, %bb20 ]
  %v3741 = phi float [ 0.000000e+00, %bb3 ], [ %v99, %bb20 ]
  %v45 = mul nuw nsw i64 %v4145, 7
  %v46 = add nuw nsw i64 %v45, 3
  %v48 = icmp ult i64 %v46, %v1
  br i1 %v48, label %bb8, label %bb34

bb34:                                             ; preds = %bb31
  tail call void @llvm.trap() #5
  unreachable

bb35:                                             ; preds = %bb8
  tail call void @llvm.trap() #5
  unreachable

bb36:                                             ; preds = %bb9
  tail call void @llvm.trap() #5
  unreachable

bb40:                                             ; preds = %__nv_sqrtf.exit11
  tail call void @llvm.trap() #5
  unreachable
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

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
