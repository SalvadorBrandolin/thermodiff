subroutine pr76_a10_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
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
   real(8) :: res, t33, t34, t35, t36, t37, t39, t40, t43, t44, t46, t47, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60
   real(8) :: t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76
   real(8) :: t1(nc), t10(nc), t11(nc), t12(nc), t15(nc), t16(nc), t17(nc), t18(nc), t2(nc), t20(nc), t21(nc), t22(nc)
   real(8) :: t26(nc), t27(nc), t28(nc), t29(nc), t30(nc), t32(nc), t38(nc), t41(nc), t42(nc), t45(nc), t48(nc), t49(nc)
   real(8) :: t5(nc), t50(nc), t6(nc), t7(nc), t8(nc), t9(nc)
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

   t39 = 0.0d0
   do k = 1, nc
      t39 = t39 + -0.45723553d0*t17(k)*t18(k)*t22(k)*t27(k)*t5(k)*t8(k)/T + 0.45723553d0* &
      ((3.0d0/2.0d0)*R**4*Tc(k)**3*t12(k)*t15(k)*t27(k)/T + (1.0d0/ &
      2.0d0)*R**4*t10(k)**3*t15(k)*t16(k)*t7(k)*t9(k)/T**2)*t17(k)*t18( &
      k)*t28(k)*t29(k)*t30(k)*t5(k)/R**4
   end do

   t40 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t40 = t40 + 0.45723553d0*((1.0d0/4.0d0)*R**4*Tc(k)*t11(l)*t12(l)*t1(k)*t1(l)*t27(k)/ &
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
      t41(j) = 0.0d0
      do k = 1, nc
         t41(j) = t41(j) + t23(j, k)
      end do
   end do

   do i = 1, nc
      t42(i) = 0.0d0
      do k = 1, nc
         t42(i) = t42(i) + t31(i, k)
      end do
   end do

   t43 = 0.0d0
   do k = 1, nc
      t43 = t43 + 0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33
   end do

   t44 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t44 = t44 + n(k)*n(l)*t3(k, l)*t4(k, l)/t33
      end do
   end do

   do i = 1, nc
      t45(i) = 0.0d0
      do k = 1, nc
         t45(i) = t45(i) + n(k)*t3(i, k)*t4(i, k)/t33
      end do
   end do

   t46 = 0.0d0
   do k = 1, nc
      t46 = t46 + -0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33**2
   end do

   t47 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t47 = t47 + -n(k)*n(l)*t3(k, l)*t4(k, l)/t33**2
      end do
   end do

   do i = 1, nc
      t48(i) = 0.0d0
      do k = 1, nc
         t48(i) = t48(i) + -n(k)*t3(i, k)*t4(i, k)/t33**2
      end do
   end do

   do j = 1, nc
      t49(j) = 0.0d0
      do k = 1, nc
         t49(j) = t49(j) + -n(k)*t3(j, k)*t4(j, k)/t33**2
      end do
   end do

   do j = 1, nc
      t50(j) = 0.0d0
      do k = 1, nc
         t50(j) = t50(j) + n(k)*t3(j, k)*t4(j, k)/t33
      end do
   end do

   t51 = 0.0d0
   do k = 1, nc
      t51 = t51 + 0.15559214d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t33**3
   end do

   t52 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t52 = t52 + 2*n(k)*n(l)*t3(k, l)*t4(k, l)/t33**3
      end do
   end do

   t53 = V - 0.414213562373095d0*t43 - 0.82842712474619d0*t44

   t54 = t43 + 2*t44

   t55 = V + 2.41421356237309d0*t43 + 4.82842712474619d0*t44

   t56 = 2*t47

   t57 = 1d0/t53

   t58 = 1d0/t54

   t59 = 1d0/V

   t60 = t53**(-2)

   t61 = t34 + 2*t35

   t62 = -t54*t59 + 1.0d0

   t63 = 1d0/t55

   t64 = log(t55*t57)

   t65 = 1d0/t62

   t66 = -t55*t60 + t57

   t67 = t54**(-2)

   t68 = V**(-2)

   t69 = t36 + 2*t37

   t70 = log(t62)

   t71 = 2*t52

   t72 = 1d0/t33

   t73 = t62**(-2)

   t74 = t55**(-2)

   t75 = t53**(-3)

   t76 = V**(-3)

   res = -R*T*t33*t70 - 0.353553390593274d0*t58*t61*t64
   f = res

   res = -R*t33*t70 - 0.353553390593274d0*t58*t64*t69
   dfdT = res

   res = -R*T*t33*t54*t65*t68 - 0.353553390593274d0*t53*t58*t61*t63*t66
   dfdV = res

   do i = 1, nc
      res = R*T*t33*t59*t65*(t46 + t56 + 2*t45(i)) - R*T*t70 - &
            0.353553390593274d0*t53*t58*t61*t63*(0.414213562373095d0*t55*t60* &
            (t46 + t56 + 2*t45(i)) + 2.41421356237309d0*t57*(t46 + t56 + 2* &
            t45(i))) - 0.707106781186548d0*t58*t64*t38(i) + &
            0.353553390593274d0*t61*t64*t67*(t46 + t56 + 2*t45(i))
      dfdn(i) = res
   end do

   res = -0.353553390593274d0*t58*t64*(t39 + 2*t40)
   dfdT2 = res

   res = 2*R*T*t33*t54*t65*t76 + R*T*t33*t54**2*t73/V**4 - &
         0.353553390593274d0*t53*t58*t61*t63*(2*t55*t75 - 2*t60) + &
         0.353553390593274d0*t53*t58*t61*t66*t74 - 0.353553390593274d0*t58 &
         *t61*t63*t66
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = R*T*t33*t59*t65*(t51 + t71 + 2*t72*t3(i, j)*t4(i, j) + 2*t48(i) + &
               2*t49(j)) + R*T*t33*t68*t73*(t46 + t56 + 2*t45(i))*(t46 + t56 + 2 &
               *t50(j)) + R*T*t59*t65*(t46 + t56 + 2*t45(i)) + R*T*t59*t65*(t46 &
               + t56 + 2*t50(j)) - 0.353553390593274d0*t53*t58*t61*t63*( &
               0.414213562373095d0*t55*t60*(t51 + t71 + 2*t72*t3(i, j)*t4(i, j) &
               + 2*t48(i) + 2*t49(j)) + 0.34314575050762d0*t55*t75*(t46 + t56 + &
               2*t45(i))*(t46 + t56 + 2*t50(j)) + 2.41421356237309d0*t57*(t51 + &
               t71 + 2*t72*t3(i, j)*t4(i, j) + 2*t48(i) + 2*t49(j)) + 2.0d0*t60* &
               (t46 + t56 + 2*t45(i))*(t46 + t56 + 2*t50(j))) + &
               0.853553390593274d0*t53*t58*t61*t74*(0.414213562373095d0*t55*t60* &
               (t46 + t56 + 2*t45(i)) + 2.41421356237309d0*t57*(t46 + t56 + 2* &
               t45(i)))*(t46 + t56 + 2*t50(j)) - 0.707106781186548d0*t53*t58*t63 &
               *(0.414213562373095d0*t55*t60*(t46 + t56 + 2*t45(i)) + &
               2.41421356237309d0*t57*(t46 + t56 + 2*t45(i)))*t41(j) - &
               0.707106781186548d0*t53*t58*t63*(0.414213562373095d0*t55*t60*(t46 &
               + t56 + 2*t50(j)) + 2.41421356237309d0*t57*(t46 + t56 + 2*t50(j &
               )))*t38(i) + 0.353553390593274d0*t53*t61*t63*t67*( &
               0.414213562373095d0*t55*t60*(t46 + t56 + 2*t45(i)) + &
               2.41421356237309d0*t57*(t46 + t56 + 2*t45(i)))*(t46 + t56 + 2*t50 &
               (j)) + 0.353553390593274d0*t53*t61*t63*t67*(0.414213562373095d0* &
               t55*t60*(t46 + t56 + 2*t50(j)) + 2.41421356237309d0*t57*(t46 + &
               t56 + 2*t50(j)))*(t46 + t56 + 2*t45(i)) + 0.146446609406726d0*t58 &
               *t61*t63*(0.414213562373095d0*t55*t60*(t46 + t56 + 2*t45(i)) + &
               2.41421356237309d0*t57*(t46 + t56 + 2*t45(i)))*(t46 + t56 + 2*t50 &
               (j)) - 0.323314343862425d0*t58*t64*t13(i, j)*t14(i, j) + &
               0.353553390593274d0*t61*t64*t67*(t51 + t71 + 2*t72*t3(i, j)*t4(i, &
               j) + 2*t48(i) + 2*t49(j)) + 0.707106781186548d0*t64*t67*(t46 + &
               t56 + 2*t45(i))*t41(j) + 0.707106781186548d0*t64*t67*(t46 + t56 + &
               2*t50(j))*t38(i) - 0.707106781186548d0*t61*t64*(t46 + t56 + 2*t45 &
               (i))*(t46 + t56 + 2*t50(j))/t54**3
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t33*t54*t65*t68 - 0.353553390593274d0*t53*t58*t63*t66*t69
   dfdTdV = res

   do i = 1, nc
      res = R*t33*t59*t65*(t46 + t56 + 2*t45(i)) - R*t70 - 0.353553390593274d0 &
            *t53*t58*t63*t69*(0.414213562373095d0*t55*t60*(t46 + t56 + 2*t45( &
            i)) + 2.41421356237309d0*t57*(t46 + t56 + 2*t45(i))) - &
            0.707106781186548d0*t58*t64*t42(i) + 0.353553390593274d0*t64*t67* &
            t69*(t46 + t56 + 2*t45(i))
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = -R*T*t33*t54*t73*t76*(t46 + t56 + 2*t45(i)) - R*T*t33*t65*t68*(t46 &
            + t56 + 2*t45(i)) - R*T*t54*t65*t68 - 0.353553390593274d0*t53*t58 &
            *t61*t63*(-0.82842712474619d0*t55*t75*(t46 + t56 + 2*t45(i)) - &
            2.0d0*t60*(t46 + t56 + 2*t45(i))) + 0.853553390593274d0*t53*t58* &
            t61*t66*t74*(t46 + t56 + 2*t45(i)) - 0.707106781186548d0*t53*t58* &
            t63*t66*t38(i) + 0.353553390593274d0*t53*t61*t63*t66*t67*(t46 + &
            t56 + 2*t45(i)) + 0.146446609406726d0*t58*t61*t63*t66*(t46 + t56 &
            + 2*t45(i))
      dfdndV(i) = res
   end do

end subroutine pr76_a10_w6
