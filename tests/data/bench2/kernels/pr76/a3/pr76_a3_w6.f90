subroutine pr76_a3_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
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
   real(8) :: res, t24, t25, t26, t29, t34, t37, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58
   real(8) :: t59, t60, t61, t62, t63
   real(8) :: t1(nc), t14(nc), t15(nc), t2(nc), t21(nc), t22(nc), t23(nc), t27(nc), t28(nc), t30(nc), t31(nc), t32(nc)
   real(8) :: t33(nc), t35(nc), t36(nc), t38(nc), t39(nc), t40(nc), t41(nc), t42(nc), t43(nc), t5(nc), t6(nc), t7(nc)
   real(8) :: t8(nc), t9(nc)
   real(8) :: t10(nc, nc), t11(nc, nc), t12(nc, nc), t13(nc, nc), t16(nc, nc), t17(nc, nc), t18(nc, nc), t19(nc, nc)
   real(8) :: t20(nc, nc), t3(nc, nc), t4(nc, nc)
   integer :: i, j, k, l, m, p, q

   do p = 1, nc
      t1(p) = 1d0/Pc(p)
   end do

   do p = 1, nc
      t2(p) = 0.038898035d0*R*Tc(p)*t1(p)
   end do

   do p = 1, nc
      do q = 1, nc
         t3(p, q) = t2(p) + t2(q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t4(p, q) = 1.0d0 - lij(p, q)
      end do
   end do

   do p = 1, nc
      t5(p) = -0.26993d0*w(p)**2 + 1.54226d0*w(p) + 0.37464d0
   end do

   do p = 1, nc
      t6(p) = sqrt(T/Tc(p))
   end do

   do p = 1, nc
      t7(p) = (1 - t6(p))*t5(p) + 1
   end do

   do p = 1, nc
      t8(p) = Tc(p)**2
   end do

   do p = 1, nc
      t9(p) = t7(p)**2
   end do

   do p = 1, nc
      do q = 1, nc
         t10(p, q) = 1.0d0 - lij(q, p)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t11(p, q) = sqrt(R**4*t1(p)*t1(q)*t8(p)*t8(q)*t9(p)*t9(q))
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t12(p, q) = 1.0d0 - kij(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t13(p, q) = 0.45723553d0*n(p)*n(q)*t11(p, q)*t12(p, q)
      end do
   end do

   do p = 1, nc
      t14(p) = Tc(p)**(-2)
   end do

   do p = 1, nc
      t15(p) = t7(p)**(-2)
   end do

   do p = 1, nc
      do q = 1, nc
         t16(p, q) = -1.0d0/2.0d0*R**4*t1(p)*t1(q)*t5(p)*t6(p)*t7(p)*t8(p)*t8(q)*t9(q)/T - &
      1.0d0/2.0d0*R**4*t1(p)*t1(q)*t5(q)*t6(q)*t7(q)*t8(p)*t8(q)*t9(p)/ &
      T
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t17(p, q) = 1.0d0 - kij(q, p)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t18(p, q) = 0.45723553d0*n(q)*t11(p, q)*t12(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t19(p, q) = 0.45723553d0*n(q)*t11(p, q)*t17(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t20(p, q) = 0.45723553d0*Pc(p)*Pc(q)*n(p)*n(q)*t11(p, q)*t12(p, q)*t14(p)*t14(q)*t15 &
      (p)*t15(q)*t16(p, q)/R**4
      end do
   end do

   do p = 1, nc
      t21(p) = Tc(p)**(-4)
   end do

   do p = 1, nc
      t22(p) = t7(p)**(-4)
   end do

   do p = 1, nc
      t23(p) = t7(p)**(-3)
   end do

   t24 = 0.0d0
   do m = 1, nc
      t24 = t24 + n(m)
   end do

   t25 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t25 = t25 + t13(k, l)
      end do
   end do

   t26 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t26 = t26 + t20(k, l)
      end do
   end do

   do i = 1, nc
      t27(i) = 0.0d0
      do l = 1, nc
         t27(i) = t27(i) + t18(i, l)
      end do
   end do

   do i = 1, nc
      t28(i) = 0.0d0
      do k = 1, nc
         t28(i) = t28(i) + t19(i, k)
      end do
   end do

   t29 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t29 = t29 + 0.45723553d0*((1.0d0/4.0d0)*R**4*Tc(k)*t1(k)*t1(l)*t5(k)**2*t8(l)*t9(l)/ &
      T + (1.0d0/4.0d0)*R**4*Tc(l)*t1(k)*t1(l)*t5(l)**2*t8(k)*t9(k)/T + &
      R**4*t1(k)*t1(l)*t5(k)*t5(l)*t6(k)*t6(l)*t7(k)*t7(l)*t8(k)*t8(l)/ &
      T**2 + (1.0d0/4.0d0)*R**4*t1(k)*t1(l)*t5(k)*t6(k)*t7(k)*t8(k)*t8( &
      l)*t9(l)/T**2 + (1.0d0/4.0d0)*R**4*t1(k)*t1(l)*t5(l)*t6(l)*t7(l)* &
      t8(k)*t8(l)*t9(k)/T**2)*Pc(k)*Pc(l)*n(k)*n(l)*t11(k, l)*t12(k, l) &
      *t14(k)*t14(l)*t15(k)*t15(l)/R**4 + 0.45723553d0*Pc(k)*Pc(l)*n(k) &
      *n(l)*t11(k, l)*t12(k, l)*t14(k)*t14(l)*t15(k)*t16(k, l)*t23(l)* &
      t5(l)*t6(l)/(R**4*T) + 0.45723553d0*Pc(k)*Pc(l)*n(k)*n(l)*t11(k, &
      l)*t12(k, l)*t14(k)*t14(l)*t15(l)*t16(k, l)*t23(k)*t5(k)*t6(k)/(R &
      **4*T) + 0.45723553d0*Pc(k)**2*Pc(l)**2*n(k)*n(l)*t11(k, l)*t12(k &
      , l)*t16(k, l)**2*t21(k)*t21(l)*t22(k)*t22(l)/R**8
      end do
   end do

   do j = 1, nc
      t30(j) = 0.0d0
      do l = 1, nc
         t30(j) = t30(j) + t18(j, l)
      end do
   end do

   do j = 1, nc
      t31(j) = 0.0d0
      do k = 1, nc
         t31(j) = t31(j) + t19(j, k)
      end do
   end do

   do i = 1, nc
      t32(i) = 0.0d0
      do k = 1, nc
         t32(i) = t32(i) + 0.45723553d0*Pc(i)*Pc(k)*n(k)*t11(i, k)*t14(i)*t14(k)*t15(i)*t15(k)*t16( &
      i, k)*t17(i, k)/R**4
      end do
   end do

   do i = 1, nc
      t33(i) = 0.0d0
      do l = 1, nc
         t33(i) = t33(i) + 0.45723553d0*Pc(i)*Pc(l)*n(l)*t11(i, l)*t12(i, l)*t14(i)*t14(l)*t15(i)* &
      t15(l)*t16(i, l)/R**4
      end do
   end do

   t34 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t34 = t34 + n(k)*n(l)*t3(k, l)*t4(k, l)/t24
      end do
   end do

   do i = 1, nc
      t35(i) = 0.0d0
      do k = 1, nc
         t35(i) = t35(i) + n(k)*t10(i, k)*t3(i, k)/t24
      end do
   end do

   do i = 1, nc
      t36(i) = 0.0d0
      do l = 1, nc
         t36(i) = t36(i) + n(l)*t3(i, l)*t4(i, l)/t24
      end do
   end do

   t37 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t37 = t37 + -n(k)*n(l)*t3(k, l)*t4(k, l)/t24**2
      end do
   end do

   do i = 1, nc
      t38(i) = 0.0d0
      do k = 1, nc
         t38(i) = t38(i) + -n(k)*t10(i, k)*t3(i, k)/t24**2
      end do
   end do

   do j = 1, nc
      t39(j) = 0.0d0
      do k = 1, nc
         t39(j) = t39(j) + -n(k)*t10(j, k)*t3(j, k)/t24**2
      end do
   end do

   do i = 1, nc
      t40(i) = 0.0d0
      do l = 1, nc
         t40(i) = t40(i) + -n(l)*t3(i, l)*t4(i, l)/t24**2
      end do
   end do

   do j = 1, nc
      t41(j) = 0.0d0
      do l = 1, nc
         t41(j) = t41(j) + -n(l)*t3(j, l)*t4(j, l)/t24**2
      end do
   end do

   do j = 1, nc
      t42(j) = 0.0d0
      do k = 1, nc
         t42(j) = t42(j) + n(k)*t10(j, k)*t3(j, k)/t24
      end do
   end do

   do j = 1, nc
      t43(j) = 0.0d0
      do l = 1, nc
         t43(j) = t43(j) + n(l)*t3(j, l)*t4(j, l)/t24
      end do
   end do

   t44 = 0.0d0
   do l = 1, nc
      do k = 1, nc
         t44 = t44 + 2*n(k)*n(l)*t3(k, l)*t4(k, l)/t24**3
      end do
   end do

   t45 = V - 0.414213562373095d0*t34

   t46 = V + 2.41421356237309d0*t34

   t47 = 1d0/t45

   t48 = 1d0/t34

   t49 = 1d0/V

   t50 = t45**(-2)

   t51 = -t34*t49 + 1.0d0

   t52 = 1d0/t46

   t53 = log(t46*t47)

   t54 = 1d0/t51

   t55 = -t46*t50 + t47

   t56 = t34**(-2)

   t57 = 1d0/t24

   t58 = V**(-2)

   t59 = log(t51)

   t60 = t51**(-2)

   t61 = t46**(-2)

   t62 = t45**(-3)

   t63 = V**(-3)

   res = -R*T*t24*t59 - 0.353553390593274d0*t25*t48*t53
   f = res

   res = -R*t24*t59 - 0.353553390593274d0*t26*t48*t53
   dfdT = res

   res = -R*T*t24*t34*t54*t58 - 0.353553390593274d0*t25*t45*t48*t52*t55
   dfdV = res

   do i = 1, nc
      res = R*T*t24*t49*t54*(t37 + t35(i) + t36(i)) - R*T*t59 - &
            0.353553390593274d0*t25*t45*t48*t52*(0.414213562373095d0*t46*t50* &
            (t37 + t35(i) + t36(i)) + 2.41421356237309d0*t47*(t37 + t35(i) + &
            t36(i))) + 0.353553390593274d0*t25*t53*t56*(t37 + t35(i) + t36(i &
            )) - 0.353553390593274d0*t48*t53*(t27(i) + t28(i))
      dfdn(i) = res
   end do

   res = -0.353553390593274d0*t29*t48*t53
   dfdT2 = res

   res = 2*R*T*t24*t34*t54*t63 + R*T*t24*t34**2*t60/V**4 - &
         0.353553390593274d0*t25*t45*t48*t52*(2*t46*t62 - 2*t50) + &
         0.353553390593274d0*t25*t45*t48*t55*t61 - 0.353553390593274d0*t25 &
         *t48*t52*t55
   dfdV2 = res

   do i = 1, nc
      do j = 1, nc
         res = R*T*t24*t49*t54*(t44 + t57*t10(i, j)*t3(i, j) + t57*t3(i, j)*t4(i, &
               j) + t38(i) + t39(j) + t40(i) + t41(j)) + R*T*t24*t58*t60*(t37 + &
               t35(i) + t36(i))*(t37 + t42(j) + t43(j)) + R*T*t49*t54*(t37 + t35 &
               (i) + t36(i)) + R*T*t49*t54*(t37 + t42(j) + t43(j)) - &
               0.353553390593274d0*t25*t45*t48*t52*(0.414213562373095d0*t46*t50* &
               (t44 + t57*t10(i, j)*t3(i, j) + t57*t3(i, j)*t4(i, j) + t38(i) + &
               t39(j) + t40(i) + t41(j)) + 0.34314575050762d0*t46*t62*(t37 + t35 &
               (i) + t36(i))*(t37 + t42(j) + t43(j)) + 2.41421356237309d0*t47*( &
               t44 + t57*t10(i, j)*t3(i, j) + t57*t3(i, j)*t4(i, j) + t38(i) + &
               t39(j) + t40(i) + t41(j)) + 2.0d0*t50*(t37 + t35(i) + t36(i))*( &
               t37 + t42(j) + t43(j))) + 0.853553390593274d0*t25*t45*t48*t61*( &
               0.414213562373095d0*t46*t50*(t37 + t35(i) + t36(i)) + &
               2.41421356237309d0*t47*(t37 + t35(i) + t36(i)))*(t37 + t42(j) + &
               t43(j)) + 0.353553390593274d0*t25*t45*t52*t56*( &
               0.414213562373095d0*t46*t50*(t37 + t35(i) + t36(i)) + &
               2.41421356237309d0*t47*(t37 + t35(i) + t36(i)))*(t37 + t42(j) + &
               t43(j)) + 0.353553390593274d0*t25*t45*t52*t56*( &
               0.414213562373095d0*t46*t50*(t37 + t42(j) + t43(j)) + &
               2.41421356237309d0*t47*(t37 + t42(j) + t43(j)))*(t37 + t35(i) + &
               t36(i)) + 0.146446609406726d0*t25*t48*t52*(0.414213562373095d0* &
               t46*t50*(t37 + t35(i) + t36(i)) + 2.41421356237309d0*t47*(t37 + &
               t35(i) + t36(i)))*(t37 + t42(j) + t43(j)) + 0.353553390593274d0* &
               t25*t53*t56*(t44 + t57*t10(i, j)*t3(i, j) + t57*t3(i, j)*t4(i, j &
               ) + t38(i) + t39(j) + t40(i) + t41(j)) - 0.707106781186548d0*t25* &
               t53*(t37 + t35(i) + t36(i))*(t37 + t42(j) + t43(j))/t34**3 - &
               0.353553390593274d0*t45*t48*t52*(0.414213562373095d0*t46*t50*(t37 &
               + t35(i) + t36(i)) + 2.41421356237309d0*t47*(t37 + t35(i) + t36(i &
               )))*(t30(j) + t31(j)) - 0.353553390593274d0*t45*t48*t52*( &
               0.414213562373095d0*t46*t50*(t37 + t42(j) + t43(j)) + &
               2.41421356237309d0*t47*(t37 + t42(j) + t43(j)))*(t27(i) + t28(i &
               )) - 0.353553390593274d0*t48*t53*(0.45723553d0*t11(i, j)*t12(i, j &
               ) + 0.45723553d0*t11(i, j)*t17(i, j)) + 0.353553390593274d0*t53* &
               t56*(t27(i) + t28(i))*(t37 + t42(j) + t43(j)) + &
               0.353553390593274d0*t53*t56*(t30(j) + t31(j))*(t37 + t35(i) + t36 &
               (i))
         dfdndn(i, j) = res
      end do
   end do

   res = -R*t24*t34*t54*t58 - 0.353553390593274d0*t26*t45*t48*t52*t55
   dfdTdV = res

   do i = 1, nc
      res = R*t24*t49*t54*(t37 + t35(i) + t36(i)) - R*t59 - &
            0.353553390593274d0*t26*t45*t48*t52*(0.414213562373095d0*t46*t50* &
            (t37 + t35(i) + t36(i)) + 2.41421356237309d0*t47*(t37 + t35(i) + &
            t36(i))) + 0.353553390593274d0*t26*t53*t56*(t37 + t35(i) + t36(i &
            )) - 0.353553390593274d0*t48*t53*(t32(i) + t33(i))
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = -R*T*t24*t34*t60*t63*(t37 + t35(i) + t36(i)) - R*T*t24*t54*t58*( &
            t37 + t35(i) + t36(i)) - R*T*t34*t54*t58 - 0.353553390593274d0* &
            t25*t45*t48*t52*(-0.82842712474619d0*t46*t62*(t37 + t35(i) + t36( &
            i)) - 2.0d0*t50*(t37 + t35(i) + t36(i))) + 0.853553390593274d0* &
            t25*t45*t48*t55*t61*(t37 + t35(i) + t36(i)) + 0.353553390593274d0 &
            *t25*t45*t52*t55*t56*(t37 + t35(i) + t36(i)) + &
            0.146446609406726d0*t25*t48*t52*t55*(t37 + t35(i) + t36(i)) - &
            0.353553390593274d0*t45*t48*t52*t55*(t27(i) + t28(i))
      dfdndV(i) = res
   end do

end subroutine pr76_a3_w6
