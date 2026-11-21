subroutine pr76_a10h_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
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
   real(8) :: res, t33, t34, t35, t36, t37, t38, t39, t40, t41, t42, t43, t45, t46, t49, t50, t52, t53, t57, t58, t59, t60
   real(8) :: t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82
   real(8) :: t83, t84, t85, t86, t87, t88
   real(8) :: t1(nc), t10(nc), t11(nc), t12(nc), t15(nc), t16(nc), t17(nc), t18(nc), t2(nc), t20(nc), t21(nc), t22(nc)
   real(8) :: t26(nc), t27(nc), t28(nc), t29(nc), t30(nc), t32(nc), t44(nc), t47(nc), t48(nc), t5(nc), t51(nc), t54(nc)
   real(8) :: t55(nc), t56(nc), t6(nc), t7(nc), t8(nc), t9(nc)
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

   t33 = R**(-8)

   t34 = R**4/T

   t35 = R**4/T**2

   t36 = R**(-4)

   t37 = 1/(R**4*T)

   t38 = 1d0/T

   t39 = 0.0d0
   do m = 1, nc
      t39 = t39 + n(m)
   end do

   t40 = 0.0d0
   do k = 1, nc
      t40 = t40 + t20(k)
   end do

   t41 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t41 = t41 + t19(k, l)
      end do
   end do

   t42 = 0.0d0
   do k = 1, nc
      t42 = t42 + t26(k)
   end do

   t43 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t43 = t43 + t25(k, l)
      end do
   end do

   do i = 1, nc
      t44(i) = 0.0d0
      do k = 1, nc
         t44(i) = t44(i) + t23(i, k)
      end do
   end do

   t45 = 0.0d0
   do k = 1, nc
      t45 = t45 + 0.45723553d0*t36*((3.0d0/2.0d0)*t34*Tc(k)**3*t12(k)*t15(k)*t27(k) + ( &
      1.0d0/2.0d0)*t35*t10(k)**3*t15(k)*t16(k)*t7(k)*t9(k))*t17(k)*t18( &
      k)*t28(k)*t29(k)*t30(k)*t5(k) - 0.45723553d0*t38*t17(k)*t18(k)* &
      t22(k)*t27(k)*t5(k)*t8(k)
   end do

   t46 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t46 = t46 + 0.45723553d0*t33*n(k)*n(l)*t13(k, l)*t14(k, l)*t24(k, l)**2*t28(k)*t28(l &
      )*t29(k)*t29(l)*t30(k)*t30(l) + 0.45723553d0*t36*((1.0d0/4.0d0)* &
      t34*Tc(k)*t11(l)*t12(l)*t1(k)*t1(l)*t27(k) + (1.0d0/4.0d0)*t34*Tc &
      (l)*t11(k)*t12(k)*t1(k)*t1(l)*t27(l) + t35*t10(k)*t10(l)*t11(k)* &
      t11(l)*t1(k)*t1(l)*t7(k)*t7(l)*t9(k)*t9(l) + (1.0d0/4.0d0)*t35* &
      t10(k)*t11(k)*t11(l)*t12(l)*t1(k)*t1(l)*t7(k)*t9(k) + (1.0d0/ &
      4.0d0)*t35*t10(l)*t11(k)*t11(l)*t12(k)*t1(k)*t1(l)*t7(l)*t9(l))* &
      Pc(k)*Pc(l)*n(k)*n(l)*t13(k, l)*t14(k, l)*t21(k)*t21(l)*t22(k)* &
      t22(l) + 0.45723553d0*t37*Pc(k)*Pc(l)*n(k)*n(l)*t13(k, l)*t14(k, &
      l)*t21(k)*t21(l)*t22(k)*t24(k, l)*t32(l)*t7(l)*t9(l) + &
      0.45723553d0*t37*Pc(k)*Pc(l)*n(k)*n(l)*t13(k, l)*t14(k, l)*t21(k) &
      *t21(l)*t22(l)*t24(k, l)*t32(k)*t7(k)*t9(k)
      end do
   end do

   do j = 1, nc
      t47(j) = 0.0d0
      do k = 1, nc
         t47(j) = t47(j) + t23(j, k)
      end do
   end do

   do i = 1, nc
      t48(i) = 0.0d0
      do k = 1, nc
         t48(i) = t48(i) + t31(i, k)
      end do
   end do

   t59 = R/t39

   t49 = 0.0d0
   do k = 1, nc
      t49 = t49 + 0.07779607d0*t59*Tc(k)*t1(k)*t5(k)*t6(k)
   end do

   t60 = 1d0/t39

   t50 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t50 = t50 + t60*n(k)*n(l)*t3(k, l)*t4(k, l)
      end do
   end do

   do i = 1, nc
      t51(i) = 0.0d0
      do k = 1, nc
         t51(i) = t51(i) + t60*n(k)*t3(i, k)*t4(i, k)
      end do
   end do

   t61 = R/t39**2

   t52 = 0.0d0
   do k = 1, nc
      t52 = t52 + -0.07779607d0*t61*Tc(k)*t1(k)*t5(k)*t6(k)
   end do

   t62 = t39**(-2)

   t53 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t53 = t53 + -t62*n(k)*n(l)*t3(k, l)*t4(k, l)
      end do
   end do

   do i = 1, nc
      t54(i) = 0.0d0
      do k = 1, nc
         t54(i) = t54(i) + -t62*n(k)*t3(i, k)*t4(i, k)
      end do
   end do

   do j = 1, nc
      t55(j) = 0.0d0
      do k = 1, nc
         t55(j) = t55(j) + -t62*n(k)*t3(j, k)*t4(j, k)
      end do
   end do

   do j = 1, nc
      t56(j) = 0.0d0
      do k = 1, nc
         t56(j) = t56(j) + t60*n(k)*t3(j, k)*t4(j, k)
      end do
   end do

   t63 = R/t39**3

   t57 = 0.0d0
   do k = 1, nc
      t57 = t57 + 0.15559214d0*t63*Tc(k)*t1(k)*t5(k)*t6(k)
   end do

   t64 = t39**(-3)

   t58 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t58 = t58 + 2*t64*n(k)*n(l)*t3(k, l)*t4(k, l)
      end do
   end do

   t65 = V - 0.414213562373095d0*t49 - 0.82842712474619d0*t50

   t66 = t49 + 2*t50

   t67 = V + 2.41421356237309d0*t49 + 4.82842712474619d0*t50

   t68 = 2*t53

   t69 = 1d0/t65

   t70 = 1d0/t66

   t71 = 1d0/V

   t72 = t65**(-2)

   t73 = t40 + 2*t41

   t74 = -t66*t71 + 1.0d0

   t75 = 1d0/t67

   t76 = log(t67*t69)

   t77 = 1d0/t74

   t78 = -t67*t72 + t69

   t79 = t66**(-2)

   t80 = V**(-2)

   t81 = t42 + 2*t43

   t82 = log(t74)

   t83 = 2*t58

   t84 = 1d0/t39

   t85 = t74**(-2)

   t86 = t67**(-2)

   t87 = t65**(-3)

   t88 = V**(-3)

   res = -R*T*t39*t82 - 0.353553390593274d0*t70*t73*t76
   f = res

   res = -R*t39*t82 - 0.353553390593274d0*t70*t76*t81
   dfdT = res

   res = -R*T*t39*t66*t77*t80 - 0.353553390593274d0*t65*t70*t73*t75*t78
   dfdV = res

   do i = 1, nc
      res = R*T*t39*t71*t77*(t52 + t68 + 2*t51(i)) - R*T*t82 - &
            0.353553390593274d0*t65*t70*t73*t75*(0.414213562373095d0*t67*t72* &
            (t52 + t68 + 2*t51(i)) + 2.41421356237309d0*t69*(t52 + t68 + 2* &
            t51(i))) - 0.707106781186548d0*t70*t76*t44(i) + &
            0.353553390593274d0*t73*t76*t79*(t52 + t68 + 2*t51(i))
      dfdn(i) = res
   end do

   res = -0.353553390593274d0*t70*t76*(t45 + 2*t46)
   dfdT2 = res

   res = 2*R*T*t39*t66*t77*t88 + R*T*t39*t66**2*t85/V**4 - &
         0.353553390593274d0*t65*t70*t73*t75*(2*t67*t87 - 2*t72) + &
         0.353553390593274d0*t65*t70*t73*t78*t86 - 0.353553390593274d0*t70 &
         *t73*t75*t78
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = R*T*t39*t71*t77*(t57 + t83 + 2*t84*t3(i, j)*t4(i, j) + 2*t54(i) + &
               2*t55(j)) + R*T*t39*t80*t85*(t52 + t68 + 2*t51(i))*(t52 + t68 + 2 &
               *t56(j)) + R*T*t71*t77*(t52 + t68 + 2*t51(i)) + R*T*t71*t77*(t52 &
               + t68 + 2*t56(j)) - 0.353553390593274d0*t65*t70*t73*t75*( &
               0.414213562373095d0*t67*t72*(t57 + t83 + 2*t84*t3(i, j)*t4(i, j) &
               + 2*t54(i) + 2*t55(j)) + 0.34314575050762d0*t67*t87*(t52 + t68 + &
               2*t51(i))*(t52 + t68 + 2*t56(j)) + 2.41421356237309d0*t69*(t57 + &
               t83 + 2*t84*t3(i, j)*t4(i, j) + 2*t54(i) + 2*t55(j)) + 2.0d0*t72* &
               (t52 + t68 + 2*t51(i))*(t52 + t68 + 2*t56(j))) + &
               0.853553390593274d0*t65*t70*t73*t86*(0.414213562373095d0*t67*t72* &
               (t52 + t68 + 2*t51(i)) + 2.41421356237309d0*t69*(t52 + t68 + 2* &
               t51(i)))*(t52 + t68 + 2*t56(j)) - 0.707106781186548d0*t65*t70*t75 &
               *(0.414213562373095d0*t67*t72*(t52 + t68 + 2*t51(i)) + &
               2.41421356237309d0*t69*(t52 + t68 + 2*t51(i)))*t47(j) - &
               0.707106781186548d0*t65*t70*t75*(0.414213562373095d0*t67*t72*(t52 &
               + t68 + 2*t56(j)) + 2.41421356237309d0*t69*(t52 + t68 + 2*t56(j &
               )))*t44(i) + 0.353553390593274d0*t65*t73*t75*t79*( &
               0.414213562373095d0*t67*t72*(t52 + t68 + 2*t51(i)) + &
               2.41421356237309d0*t69*(t52 + t68 + 2*t51(i)))*(t52 + t68 + 2*t56 &
               (j)) + 0.353553390593274d0*t65*t73*t75*t79*(0.414213562373095d0* &
               t67*t72*(t52 + t68 + 2*t56(j)) + 2.41421356237309d0*t69*(t52 + &
               t68 + 2*t56(j)))*(t52 + t68 + 2*t51(i)) + 0.146446609406726d0*t70 &
               *t73*t75*(0.414213562373095d0*t67*t72*(t52 + t68 + 2*t51(i)) + &
               2.41421356237309d0*t69*(t52 + t68 + 2*t51(i)))*(t52 + t68 + 2*t56 &
               (j)) - 0.323314343862425d0*t70*t76*t13(i, j)*t14(i, j) + &
               0.353553390593274d0*t73*t76*t79*(t57 + t83 + 2*t84*t3(i, j)*t4(i, &
               j) + 2*t54(i) + 2*t55(j)) + 0.707106781186548d0*t76*t79*(t52 + &
               t68 + 2*t51(i))*t47(j) + 0.707106781186548d0*t76*t79*(t52 + t68 + &
               2*t56(j))*t44(i) - 0.707106781186548d0*t73*t76*(t52 + t68 + 2*t51 &
               (i))*(t52 + t68 + 2*t56(j))/t66**3
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t39*t66*t77*t80 - 0.353553390593274d0*t65*t70*t75*t78*t81
   dfdTdV = res

   do i = 1, nc
      res = R*t39*t71*t77*(t52 + t68 + 2*t51(i)) - R*t82 - 0.353553390593274d0 &
            *t65*t70*t75*t81*(0.414213562373095d0*t67*t72*(t52 + t68 + 2*t51( &
            i)) + 2.41421356237309d0*t69*(t52 + t68 + 2*t51(i))) - &
            0.707106781186548d0*t70*t76*t48(i) + 0.353553390593274d0*t76*t79* &
            t81*(t52 + t68 + 2*t51(i))
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = -R*T*t39*t66*t85*t88*(t52 + t68 + 2*t51(i)) - R*T*t39*t77*t80*(t52 &
            + t68 + 2*t51(i)) - R*T*t66*t77*t80 - 0.353553390593274d0*t65*t70 &
            *t73*t75*(-0.82842712474619d0*t67*t87*(t52 + t68 + 2*t51(i)) - &
            2.0d0*t72*(t52 + t68 + 2*t51(i))) + 0.853553390593274d0*t65*t70* &
            t73*t78*t86*(t52 + t68 + 2*t51(i)) - 0.707106781186548d0*t65*t70* &
            t75*t78*t44(i) + 0.353553390593274d0*t65*t73*t75*t78*t79*(t52 + &
            t68 + 2*t51(i)) + 0.146446609406726d0*t70*t73*t75*t78*(t52 + t68 &
            + 2*t51(i))
      dfdndV(i) = res
   end do

end subroutine pr76_a10h_w6
