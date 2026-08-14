; ModuleID = '/home/kranex/dev/rust-metaprogramming/examples/oxide/nearest_neighbor.linked.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@.str = private unnamed_addr constant [11 x i8] c"__CUDA_FTZ\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"__CUDA_PREC_SQRT\00", align 1
@llvm.used = appending global [1 x ptr] [ptr @euclid], section "llvm.metadata"

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @euclid(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(address_is_null) %v2, i64 %v3, float %v4, float %v5) #0 {
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
  %v17.i11.i.not4 = or i1 %v14.i.i, %v16.i.i
  %v5.i12.i = icmp ne i32 %v4.i9.i, 1
  %v7.i13.i = icmp ne i32 %v6.i10.i, 1
  %v8.not.not.i.i.not3 = or i1 %v5.i12.i, %v7.i13.i
  %.v18.i.i.not2 = or i1 %v8.not.not.i.i.not3, %v17.i11.i.not4
  %v19.not.i = icmp uge i64 %v18.i.i, %v3
  %v23.i = getelementptr inbounds nuw float, ptr %v2, i64 %v18.i.i
  %.not = select i1 %.v18.i.i.not2, i1 true, i1 %v19.not.i
  %v21.not.not1 = icmp eq ptr %v2, null
  %v21.not.not = select i1 %.not, i1 true, i1 %v21.not.not1
  br i1 %v21.not.not, label %bb7, label %bb3

bb3:                                              ; preds = %entry
  %0 = shl nuw nsw i64 %v18.i.i, 1
  %v36 = icmp ult i64 %0, %v1
  br i1 %v36, label %bb4, label %bb10

bb4:                                              ; preds = %bb3
  %v41 = or disjoint i64 %0, 1
  %v42 = icmp ult i64 %v41, %v1
  br i1 %v42, label %bb5, label %bb11

bb5:                                              ; preds = %bb4
  %v38 = getelementptr inbounds nuw float, ptr %v0, i64 %0
  %v39 = load float, ptr %v38, align 4
  %v40 = fsub contract float %v4, %v39
  %v44 = getelementptr inbounds nuw float, ptr %v0, i64 %v41
  %v45 = load float, ptr %v44, align 4
  %v46 = fsub contract float %v5, %v45
  %v47 = fmul contract float %v40, %v40
  %v48 = fmul contract float %v46, %v46
  %v49 = fadd contract float %v47, %v48
  %1 = tail call i32 @__nvvm_reflect(ptr nonnull @.str) #6
  %.not.i = icmp eq i32 %1, 0
  %2 = tail call i32 @__nvvm_reflect(ptr nonnull @.str.2) #6
  %.not1.i = icmp eq i32 %2, 0
  br i1 %.not.i, label %8, label %3

3:                                                ; preds = %bb5
  br i1 %.not1.i, label %6, label %4

4:                                                ; preds = %3
  %5 = tail call float @llvm.nvvm.sqrt.rn.ftz.f(float %v49) #6
  br label %__nv_sqrtf.exit

6:                                                ; preds = %3
  %7 = tail call float @llvm.nvvm.sqrt.approx.ftz.f(float %v49) #6
  br label %__nv_sqrtf.exit

8:                                                ; preds = %bb5
  br i1 %.not1.i, label %11, label %9

9:                                                ; preds = %8
  %10 = tail call float @llvm.nvvm.sqrt.rn.f(float %v49) #6
  br label %__nv_sqrtf.exit

11:                                               ; preds = %8
  %12 = tail call float @llvm.nvvm.sqrt.approx.f(float %v49) #6
  br label %__nv_sqrtf.exit

__nv_sqrtf.exit:                                  ; preds = %4, %6, %9, %11
  %.0.i = phi float [ %5, %4 ], [ %7, %6 ], [ %10, %9 ], [ %12, %11 ]
  store float %.0.i, ptr %v23.i, align 4
  br label %bb7

bb7:                                              ; preds = %entry, %__nv_sqrtf.exit
  ret void

bb10:                                             ; preds = %bb3
  tail call void @llvm.trap() #5
  unreachable

bb11:                                             ; preds = %bb4
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
