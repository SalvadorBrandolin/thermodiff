subroutine pr76_a6_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
     dfdndV)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), intent(in) :: Pc(:)
   real(8), intent(in) :: Tc(:)
   real(8), intent(in) :: kij(:, :)
   real(8), intent(in) :: lij(:, :)
   real(8), intent(in) :: n(:)
   real(8), intent(in) :: w(:)
   real(8), intent(out) :: f
   real(8), intent(out) :: dfdT
   real(8), intent(out) :: dfdV
   real(8), intent(out) :: dfdn(:)
   real(8), intent(out) :: dfdT2
   real(8), intent(out) :: dfdV2
   real(8), intent(out) :: dfdndn(:, :)
   real(8), intent(out) :: dfdTdV
   real(8), intent(out) :: dfdndT(:)
   real(8), intent(out) :: dfdndV(:)
   real(8) :: res, t33, t34, t35, t36, t37, t40, t41, t46, t47, t50, t51, t58, t59, t60, t61, t62, t63, t64, t65, t66, t67
   real(8) :: t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82, t83
   real(8) :: t1(nc), t10(nc), t11(nc), t12(nc), t15(nc), t16(nc), t17(nc), t18(nc), t2(nc), t20(nc), t21(nc), t22(nc)
   real(8) :: t26(nc), t27(nc), t28(nc), t29(nc), t30(nc), t32(nc), t38(nc), t39(nc), t42(nc), t43(nc), t44(nc), t45(nc)
   real(8) :: t48(nc), t49(nc), t5(nc), t52(nc), t53(nc), t54(nc), t55(nc), t56(nc), t57(nc), t6(nc), t7(nc), t8(nc)
   real(8) :: t9(nc)
   real(8) :: t13(nc, nc), t14(nc, nc), t19(nc, nc), t23(nc, nc), t24(nc, nc), t25(nc, nc), t3(nc, nc), t31(nc, nc)
   real(8) :: t4(nc, nc)
   integer :: i, j, k, l, m, p, q

   do p = 1, nc
      t1(p) = 1d0/Pc(p)
   end do

   do p = 1, nc
      t2(p) = 0.038898035d0*R*Tc(p)*t1(p)
   end do

   do p = 1, nc
      do q = p, nc
         t3(p, q) = 1.0d0 - lij(p, q)
         t3(q, p) = t3(p, q)
      end do
   end do

   do p = 1, nc
      do q = p, nc
         t4(p, q) = t2(p) + t2(q)
         t4(q, p) = t4(p, q)
      end do
   end do

   do p = 1, nc
      t5(p) = n(p)**2
   end do

   do p = 1, nc
      t6(p) = 1.0d0 - lij(p, p)
   end do

   do p = 1, nc
      t7(p) = -0.26993d0*w(p)**2 + 1.54226d0*w(p) + 0.37464d0
   end do

   do p = 1, nc
      t8(p) = 1d0/Tc(p)
   end do

   do p = 1, nc
      t9(p) = sqrt(T*t8(p))
   end do

   do p = 1, nc
      t10(p) = (1 - t9(p))*t7(p) + 1
   end do

   do p = 1, nc
      t11(p) = Tc(p)**2
   end do

   do p = 1, nc
      t12(p) = t10(p)**2
   end do

   do p = 1, nc
      do q = p, nc
         t13(p, q) = sqrt(R**4*t11(p)*t11(q)*t12(p)*t12(q)*t1(p)*t1(q))
         t13(q, p) = t13(p, q)
      end do
   end do

   do p = 1, nc
      do q = p, nc
         t14(p, q) = 1.0d0 - kij(p, q)
         t14(q, p) = t14(p, q)
      end do
   end do

   do p = 1, nc
      t15(p) = Pc(p)**(-2)
   end do

   do p = 1, nc
      t16(p) = Tc(p)**4
   end do

   do p = 1, nc
      t17(p) = sqrt(R**4*t10(p)**4*t15(p)*t16(p))
   end do

   do p = 1, nc
      t18(p) = 1.0d0 - kij(p, p)
   end do

   do p = 1, nc
      do q = 1, nc
         t19(p, q) = 0.45723553d0*n(p)*n(q)*t13(p, q)*t14(p, q)
      end do
   end do

   do p = 1, nc
      t20(p) = 0.45723553d0*t17(p)*t18(p)*t5(p)
   end do

   do p = 1, nc
      t21(p) = Tc(p)**(-2)
   end do

   do p = 1, nc
      t22(p) = t10(p)**(-2)
   end do

   do p = 1, nc
      do q = 1, nc
         t23(p, q) = 0.45723553d0*n(q)*t13(p, q)*t14(p, q)
      end do
   end do

   do p = 1, nc
      do q = p, nc
         t24(p, q) = -1.0d0/2.0d0*R**4*t10(p)*t11(p)*t11(q)*t12(q)*t1(p)*t1(q)*t7(p)*t9(p)/T &
      - 1.0d0/2.0d0*R**4*t10(q)*t11(p)*t11(q)*t12(p)*t1(p)*t1(q)*t7(q)* &
      t9(q)/T
         t24(q, p) = t24(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t25(p, q) = 0.45723553d0*Pc(p)*Pc(q)*n(p)*n(q)*t13(p, q)*t14(p, q)*t21(p)*t21(q)*t22 &
      (p)*t22(q)*t24(p, q)/R**4
      end do
   end do

   do p = 1, nc
      t26(p) = -0.45723553d0*t17(p)*t18(p)*t5(p)*t7(p)*t9(p)/(T*t10(p))
   end do

   do p = 1, nc
      t27(p) = t7(p)**2
   end do

   do p = 1, nc
      t28(p) = Tc(p)**(-4)
   end do

   do p = 1, nc
      t29(p) = t10(p)**(-4)
   end do

   do p = 1, nc
      t30(p) = Pc(p)**2
   end do

   do p = 1, nc
      do q = 1, nc
         t31(p, q) = 0.45723553d0*Pc(p)*Pc(q)*n(q)*t13(p, q)*t14(p, q)*t21(p)*t21(q)*t22(p)* &
      t22(q)*t24(p, q)/R**4
      end do
   end do

   do p = 1, nc
      t32(p) = t10(p)**(-3)
   end do

   t33 = 0.0d0
   do m = 1, nc
      t33 = t33 + n(m)
   end do

   t34 = 0.0d0
   do k = 1, nc
      t34 = t34 + t20(k)
   end do

   t35 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t35 = t35 + t19(k, l)
      end do
   end do

   t36 = 0.0d0
   do k = 1, nc
      t36 = t36 + t26(k)
   end do

   t37 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t37 = t37 + t25(k, l)
      end do
   end do

   do i = 1, nc
      t38(i) = 0.0d0
      do k = 1, nc
         t38(i) = t38(i) + t23(i, k)
      end do
   end do

   do i = 1, nc
      t39(i) = 0.0d0
      do l = 1, nc
         t39(i) = t39(i) + t23(i, l)
      end do
   end do

   t40 = 0.0d0
   do k = 1, nc
      t40 = t40 + -0.45723553d0*t17(k)*t18(k)*t22(k)*t27(k)*t5(k)*t8(k)/T + 0.45723553d0* &
      ((3.0d0/2.0d0)*R**4*Tc(k)**3*t12(k)*t15(k)*t27(k)/T + (1.0d0/ &
      2.0d0)*R**4*t10(k)**3*t15(k)*t16(k)*t7(k)*t9(k)/T**2)*t17(k)*t18( &
      k)*t28(k)*t29(k)*t30(k)*t5(k)/R**4
   end do

   t41 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t41 = t41 + 0.45723553d0*((1.0d0/4.0d0)*R**4*Tc(k)*t11(l)*t12(l)*t1(k)*t1(l)*t27(k)/ &
      T + (1.0d0/4.0d0)*R**4*Tc(l)*t11(k)*t12(k)*t1(k)*t1(l)*t27(l)/T + &
      R**4*t10(k)*t10(l)*t11(k)*t11(l)*t1(k)*t1(l)*t7(k)*t7(l)*t9(k)*t9 &
      (l)/T**2 + (1.0d0/4.0d0)*R**4*t10(k)*t11(k)*t11(l)*t12(l)*t1(k)* &
      t1(l)*t7(k)*t9(k)/T**2 + (1.0d0/4.0d0)*R**4*t10(l)*t11(k)*t11(l)* &
      t12(k)*t1(k)*t1(l)*t7(l)*t9(l)/T**2)*Pc(k)*Pc(l)*n(k)*n(l)*t13(k, &
      l)*t14(k, l)*t21(k)*t21(l)*t22(k)*t22(l)/R**4 + 0.45723553d0*Pc(k &
      )*Pc(l)*n(k)*n(l)*t13(k, l)*t14(k, l)*t21(k)*t21(l)*t22(k)*t24(k, &
      l)*t32(l)*t7(l)*t9(l)/(R**4*T) + 0.45723553d0*Pc(k)*Pc(l)*n(k)*n( &
      l)*t13(k, l)*t14(k, l)*t21(k)*t21(l)*t22(l)*t24(k, l)*t32(k)*t7(k &
      )*t9(k)/(R**4*T) + 0.45723553d0*n(k)*n(l)*t13(k, l)*t14(k, l)*t24 &
      (k, l)**2*t28(k)*t28(l)*t29(k)*t29(l)*t30(k)*t30(l)/R**8
      end do
   end do

   do j = 1, nc
      t42(j) = 0.0d0
      do k = 1, nc
         t42(j) = t42(j) + t23(j, k)
      end do
   end do

   do j = 1, nc
      t43(j) = 0.0d0
      do l = 1, nc
         t43(j) = t43(j) + t23(j, l)
      end do
   end do

   do i = 1, nc
      t44(i) = 0.0d0
      do k = 1, nc
         t44(i) = t44(i) + t31(i, k)
      end do
   end do

   do i = 1, nc
      t45(i) = 0.0d0
      do l = 1, nc
         t45(i) = t45(i) + t31(i, l)
      end do
   end do

   t46 = 0.0d0
   do k = 1, nc
      t46 = t46 + 0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33
   end do

   t47 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t47 = t47 + n(k)*n(l)*t3(k, l)*t4(k, l)/t33
      end do
   end do

   do i = 1, nc
      t48(i) = 0.0d0
      do k = 1, nc
         t48(i) = t48(i) + n(k)*t3(i, k)*t4(i, k)/t33
      end do
   end do

   do i = 1, nc
      t49(i) = 0.0d0
      do l = 1, nc
         t49(i) = t49(i) + n(l)*t3(i, l)*t4(i, l)/t33
      end do
   end do

   t50 = 0.0d0
   do k = 1, nc
      t50 = t50 + -0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33**2
   end do

   t51 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t51 = t51 + -n(k)*n(l)*t3(k, l)*t4(k, l)/t33**2
      end do
   end do

   do i = 1, nc
      t52(i) = 0.0d0
      do k = 1, nc
         t52(i) = t52(i) + -n(k)*t3(i, k)*t4(i, k)/t33**2
      end do
   end do

   do j = 1, nc
      t53(j) = 0.0d0
      do k = 1, nc
         t53(j) = t53(j) + -n(k)*t3(j, k)*t4(j, k)/t33**2
      end do
   end do

   do i = 1, nc
      t54(i) = 0.0d0
      do l = 1, nc
         t54(i) = t54(i) + -n(l)*t3(i, l)*t4(i, l)/t33**2
      end do
   end do

   do j = 1, nc
      t55(j) = 0.0d0
      do l = 1, nc
         t55(j) = t55(j) + -n(l)*t3(j, l)*t4(j, l)/t33**2
      end do
   end do

   do j = 1, nc
      t56(j) = 0.0d0
      do k = 1, nc
         t56(j) = t56(j) + n(k)*t3(j, k)*t4(j, k)/t33
      end do
   end do

   do j = 1, nc
      t57(j) = 0.0d0
      do l = 1, nc
         t57(j) = t57(j) + n(l)*t3(j, l)*t4(j, l)/t33
      end do
   end do

   t58 = 0.0d0
   do k = 1, nc
      t58 = t58 + 0.15559214d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33**3
   end do

   t59 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t59 = t59 + 2*n(k)*n(l)*t3(k, l)*t4(k, l)/t33**3
      end do
   end do

   t60 = V - 0.414213562373095d0*t46 - 0.82842712474619d0*t47

   t61 = t46 + 2*t47

   t62 = V + 2.41421356237309d0*t46 + 4.82842712474619d0*t47

   t63 = 2*t51

   t64 = 1d0/t60

   t65 = 1d0/t61

   t66 = 1d0/V

   t67 = t60**(-2)

   t68 = t34 + 2*t35

   t69 = -t61*t66 + 1.0d0

   t70 = 1d0/t62

   t71 = log(t62*t64)

   t72 = 1d0/t69

   t73 = -t62*t67 + t64

   t74 = t61**(-2)

   t75 = V**(-2)

   t76 = t36 + 2*t37

   t77 = log(t69)

   t78 = 2*t59

   t79 = 1d0/t33

   t80 = t69**(-2)

   t81 = t62**(-2)

   t82 = t60**(-3)

   t83 = V**(-3)

   res = -R*T*t33*t77 - 0.353553390593274d0*t65*t68*t71
   f = res

   res = -R*t33*t77 - 0.353553390593274d0*t65*t71*t76
   dfdT = res

   res = -R*T*t33*t61*t72*t75 - 0.353553390593274d0*t60*t65*t68*t70*t73
   dfdV = res

   do i = 1, nc
      res = R*T*t33*t66*t72*(t50 + t63 + t48(i) + t49(i)) - R*T*t77 - &
            0.353553390593274d0*t60*t65*t68*t70*(0.414213562373095d0*t62*t67* &
            (t50 + t63 + t48(i) + t49(i)) + 2.41421356237309d0*t64*(t50 + t63 &
            + t48(i) + t49(i))) - 0.353553390593274d0*t65*t71*(t38(i) + t39(i &
            )) + 0.353553390593274d0*t68*t71*t74*(t50 + t63 + t48(i) + t49(i &
            ))
      dfdn(i) = res
   end do

   res = -0.353553390593274d0*t65*t71*(t40 + 2*t41)
   dfdT2 = res

   res = 2*R*T*t33*t61*t72*t83 + R*T*t33*t61**2*t80/V**4 - &
         0.353553390593274d0*t60*t65*t68*t70*(2*t62*t82 - 2*t67) + &
         0.353553390593274d0*t60*t65*t68*t73*t81 - 0.353553390593274d0*t65 &
         *t68*t70*t73
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = R*T*t33*t66*t72*(t58 + t78 + 2*t79*t3(i, j)*t4(i, j) + t52(i) + &
               t53(j) + t54(i) + t55(j)) + R*T*t33*t75*t80*(t50 + t63 + t48(i) + &
               t49(i))*(t50 + t63 + t56(j) + t57(j)) + R*T*t66*t72*(t50 + t63 + &
               t48(i) + t49(i)) + R*T*t66*t72*(t50 + t63 + t56(j) + t57(j)) - &
               0.353553390593274d0*t60*t65*t68*t70*(0.414213562373095d0*t62*t67* &
               (t58 + t78 + 2*t79*t3(i, j)*t4(i, j) + t52(i) + t53(j) + t54(i) + &
               t55(j)) + 0.34314575050762d0*t62*t82*(t50 + t63 + t48(i) + t49(i &
               ))*(t50 + t63 + t56(j) + t57(j)) + 2.41421356237309d0*t64*(t58 + &
               t78 + 2*t79*t3(i, j)*t4(i, j) + t52(i) + t53(j) + t54(i) + t55(j &
               )) + 2.0d0*t67*(t50 + t63 + t48(i) + t49(i))*(t50 + t63 + t56(j) &
               + t57(j))) + 0.853553390593274d0*t60*t65*t68*t81*( &
               0.414213562373095d0*t62*t67*(t50 + t63 + t48(i) + t49(i)) + &
               2.41421356237309d0*t64*(t50 + t63 + t48(i) + t49(i)))*(t50 + t63 &
               + t56(j) + t57(j)) - 0.353553390593274d0*t60*t65*t70*( &
               0.414213562373095d0*t62*t67*(t50 + t63 + t48(i) + t49(i)) + &
               2.41421356237309d0*t64*(t50 + t63 + t48(i) + t49(i)))*(t42(j) + &
               t43(j)) - 0.353553390593274d0*t60*t65*t70*(0.414213562373095d0* &
               t62*t67*(t50 + t63 + t56(j) + t57(j)) + 2.41421356237309d0*t64*( &
               t50 + t63 + t56(j) + t57(j)))*(t38(i) + t39(i)) + &
               0.353553390593274d0*t60*t68*t70*t74*(0.414213562373095d0*t62*t67* &
               (t50 + t63 + t48(i) + t49(i)) + 2.41421356237309d0*t64*(t50 + t63 &
               + t48(i) + t49(i)))*(t50 + t63 + t56(j) + t57(j)) + &
               0.353553390593274d0*t60*t68*t70*t74*(0.414213562373095d0*t62*t67* &
               (t50 + t63 + t56(j) + t57(j)) + 2.41421356237309d0*t64*(t50 + t63 &
               + t56(j) + t57(j)))*(t50 + t63 + t48(i) + t49(i)) + &
               0.146446609406726d0*t65*t68*t70*(0.414213562373095d0*t62*t67*(t50 &
               + t63 + t48(i) + t49(i)) + 2.41421356237309d0*t64*(t50 + t63 + &
               t48(i) + t49(i)))*(t50 + t63 + t56(j) + t57(j)) - &
               0.323314343862425d0*t65*t71*t13(i, j)*t14(i, j) + &
               0.353553390593274d0*t68*t71*t74*(t58 + t78 + 2*t79*t3(i, j)*t4(i, &
               j) + t52(i) + t53(j) + t54(i) + t55(j)) + 0.353553390593274d0*t71 &
               *t74*(t38(i) + t39(i))*(t50 + t63 + t56(j) + t57(j)) + &
               0.353553390593274d0*t71*t74*(t42(j) + t43(j))*(t50 + t63 + t48(i &
               ) + t49(i)) - 0.707106781186548d0*t68*t71*(t50 + t63 + t48(i) + &
               t49(i))*(t50 + t63 + t56(j) + t57(j))/t61**3
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t33*t61*t72*t75 - 0.353553390593274d0*t60*t65*t70*t73*t76
   dfdTdV = res

   do i = 1, nc
      res = R*t33*t66*t72*(t50 + t63 + t48(i) + t49(i)) - R*t77 - &
            0.353553390593274d0*t60*t65*t70*t76*(0.414213562373095d0*t62*t67* &
            (t50 + t63 + t48(i) + t49(i)) + 2.41421356237309d0*t64*(t50 + t63 &
            + t48(i) + t49(i))) - 0.353553390593274d0*t65*t71*(t44(i) + t45(i &
            )) + 0.353553390593274d0*t71*t74*t76*(t50 + t63 + t48(i) + t49(i &
            ))
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = -R*T*t33*t61*t80*t83*(t50 + t63 + t48(i) + t49(i)) - R*T*t33*t72* &
            t75*(t50 + t63 + t48(i) + t49(i)) - R*T*t61*t72*t75 - &
            0.353553390593274d0*t60*t65*t68*t70*(-0.82842712474619d0*t62*t82* &
            (t50 + t63 + t48(i) + t49(i)) - 2.0d0*t67*(t50 + t63 + t48(i) + &
            t49(i))) + 0.853553390593274d0*t60*t65*t68*t73*t81*(t50 + t63 + &
            t48(i) + t49(i)) - 0.353553390593274d0*t60*t65*t70*t73*(t38(i) + &
            t39(i)) + 0.353553390593274d0*t60*t68*t70*t73*t74*(t50 + t63 + &
            t48(i) + t49(i)) + 0.146446609406726d0*t65*t68*t70*t73*(t50 + t63 &
            + t48(i) + t49(i))
      dfdndV(i) = res
   end do

end subroutine pr76_a6_w6
