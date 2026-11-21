subroutine pr76_c6m_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
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
   real(8) :: res, t1, t11, t12, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29
   real(8) :: t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t4, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49
   real(8) :: t5, t50, t51, t52, t54, t55, t56, t57, t58, t6, t61, t62, t63, t64, t65, t66, t67, t68, t69, t9
   real(8) :: cmp0(nc), cmp0_T(nc), cmp0_TT(nc), cmp1(nc), cmp1_T(nc), cmp1_TT(nc), t10(nc), t53(nc), t59(nc), t60(nc)
   real(8) :: t7(nc), t8(nc)
   real(8) :: mat2(nc, nc)
   integer :: i, ia, ib, ic, j, k, l, m, p

   do ic = 1, nc
      cmp0(ic) = ((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0 &
      ) + 1)**2*Tc(ic)**2/Pc(ic)
   end do

   t51 = 1d0/T

   do ic = 1, nc
      cmp0_T(ic) = -t51*sqrt(T/Tc(ic))*((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + &
      1.54226d0*w(ic) + 0.37464d0) + 1)*(-0.26993d0*w(ic)**2 + &
      1.54226d0*w(ic) + 0.37464d0)*Tc(ic)**2/Pc(ic)
   end do

   t52 = T**(-2)

   do ic = 1, nc
      cmp0_TT(ic) = (1.0d0/2.0d0)*t51*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0)**2 &
      *Tc(ic)/Pc(ic) + (1.0d0/2.0d0)*t52*sqrt(T/Tc(ic))*((1 - sqrt(T/Tc &
      (ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0) + 1)*( &
      -0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0)*Tc(ic)**2/Pc( &
      ic)
   end do

   do ic = 1, nc
      cmp1(ic) = sqrt(cmp0(ic))
   end do

   do ic = 1, nc
      cmp1_T(ic) = (1.0d0/2.0d0)*cmp0_T(ic)*cmp1(ic)/cmp0(ic)
   end do

   do ic = 1, nc
      cmp1_TT(ic) = ((1.0d0/2.0d0)*cmp0_TT(ic)/cmp0(ic) - 1.0d0/2.0d0*cmp0_T(ic)**2/cmp0(ic) &
      **2)*cmp1(ic) + (1.0d0/2.0d0)*cmp0_T(ic)*cmp1_T(ic)/cmp0(ic)
   end do

   do p = 1, nc
      t53(p) = 0.038898035d0*R*Tc(p)/Pc(p)
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat2(ia, ib) = (1.0d0 - lij(ia, ib))*(t53(ia) + t53(ib))
         mat2(ib, ia) = mat2(ia, ib)
      end do
   end do

   do l = 1, nc
      t7(l) = 0.0d0
   end do
   do l = 1, nc
      t67 = 0.0d0
      do k = 1, l - 1
         t7(k) = t7(k) + mat2(k, l)*n(l)
         t67 = t67 + mat2(k, l)*n(k)
      end do
      t7(l) = t7(l) + t67 + mat2(l, l)*n(l)
   end do

   do l = 1, nc
      t8(l) = 0.0d0
   end do
   do l = 1, nc
      t68 = 0.0d0
      do k = 1, l - 1
         t8(k) = t8(k) + (1.0d0 - kij(k, l))*cmp1(l)*n(l)
         t68 = t68 + (1.0d0 - kij(k, l))*cmp1(k)*n(k)
      end do
      t8(l) = t8(l) + t68 + (1.0d0 - kij(l, l))*cmp1(l)*n(l)
   end do

   t9 = 0.0d0
   do m = 1, nc
      t9 = t9 + n(m)
   end do

   do l = 1, nc
      t10(l) = 0.0d0
   end do
   do l = 1, nc
      t69 = 0.0d0
      do k = 1, l - 1
         t10(k) = t10(k) + (1.0d0 - kij(k, l))*cmp1_T(l)*n(l)
         t69 = t69 + (1.0d0 - kij(k, l))*cmp1_T(k)*n(k)
      end do
      t10(l) = t10(l) + t69 + (1.0d0 - kij(l, l))*cmp1_T(l)*n(l)
   end do

   t11 = 0.0d0
   do l = 1, nc
      t11 = t11 + n(l)*t7(l)
   end do

   t12 = 0.0d0
   do l = 1, nc
      t12 = t12 + cmp1(l)*n(l)*t8(l)
   end do

   t13 = 0.0d0
   do k = 1, nc
      t13 = t13 + cmp1_T(k)*n(k)*t8(k)
   end do

   t14 = 0.0d0
   do k = 1, nc
      t14 = t14 + cmp1_TT(k)*n(k)*t8(k)
   end do

   t15 = 0.0d0
   do l = 1, nc
      t15 = t15 + cmp1_T(l)*n(l)*t10(l)
   end do

   t16 = 1d0/t9

   t17 = V - 0.414213562373095d0*t11*t16

   t18 = V + 2.41421356237309d0*t11*t16

   t19 = 1d0/t17

   t20 = sqrt(R**4)

   t21 = 1d0/t11

   t22 = t17**(-2)

   t23 = t9**(-2)

   t24 = 1d0/t18

   t25 = 1d0/V

   t26 = -t11*t16*t25 + 1.0d0

   t27 = log(t18*t19)

   t28 = 0.414213562373095d0*t16*t18*t22 + 2.41421356237309d0*t16*t19

   t29 = -0.414213562373095d0*t11*t18*t22*t23 - 2.41421356237309d0*t11*t19*t23

   t30 = t11**(-2)

   t31 = 1d0/t26

   t32 = t9**(-3)

   t33 = t26**(-2)

   t34 = t18**(-2)

   t35 = V**(-2)

   t36 = t17**(-3)

   t37 = -t18*t22 + t19

   t38 = t11**2

   t39 = log(t26)

   t40 = -0.161657171931213d0*t17*t21*t24*t29*t9 - 0.161657171931213d0*t21*t27

   t41 = -0.161657171931213d0*t17*t21*t24*t28*t9 + 0.161657171931213d0*t27*t30*t9

   t42 = V**(-3)

   t43 = -0.161657171931213d0*t12*t17*t20*t21*t24*t9*(-0.34314575050762d0*t11*t18 &
      *t32*t36 - 2.0d0*t11*t22*t32 - 0.414213562373095d0*t18*t22*t23 - &
      2.41421356237309d0*t19*t23)

   t44 = R*T*t25*t31 - 0.161657171931213d0*t12*t17*t20*t21*t24*t28*t9 + &
      0.161657171931213d0*t12*t20*t27*t30*t9

   t45 = 0.161657171931213d0*t12*t17*t20*t24*t29*t30*t9

   t46 = -0.161657171931213d0*t12*t17*t20*t21*t24*t28

   t47 = -R*T*t11*t23*t33*t35

   t48 = 0.161657171931213d0*t12*t20*t27*t30

   t49 = -t18*t19 + 1

   t50 = t9**(-4)

   t54 = t20*t21*t27*t9

   t55 = -R*T*t11*t16*t25*t31 - R*T*t39 - 0.161657171931213d0*t12*t17*t20*t21*t24 &
      *t29*t9 - 0.161657171931213d0*t12*t20*t21*t27

   t56 = 4*R*T*t16*t33*t35 - 0.64662868772485d0*t12*t17*t20*t21*t24*t9*( &
      0.34314575050762d0*t18*t23*t36 + 2.0d0*t22*t23) + &
      1.56109974772485d0*t12*t17*t20*t21*t28*t34 + 1.2932573754497d0* &
      t12*t17*t20*t24*t28*t30*t9 + 0.26784237227515d0*t12*t20*t21*t24* &
      t28 - 1.2932573754497d0*t12*t20*t27*t9/t11**3

   t57 = t20*t41

   t58 = R*T*t32*t33*t35*t38 - 0.390274936931213d0*t12*t16*t17*t20*t29*t34 - &
      0.0669605930687875d0*t12*t16*t20*t24*t29 - 0.323314343862425d0* &
      t12*t17*t20*t21*t24*t29 - 0.161657171931213d0*t12*t17*t20*t21*t24 &
      *t9*(0.82842712474619d0*t11*t18*t22*t32 + 4.82842712474619d0*t11* &
      t19*t32 + 0.34314575050762d0*t18*t36*t38*t50 + 2.0d0*t22*t38*t50)

   do p = 1, nc
      t59(p) = 2*t20*t40*cmp1(p)*t8(p) + (-0.780549873862425d0*t12*t16*t17*t20*t28*t34 &
      - 0.133921186137575d0*t12*t16*t20*t24*t28 + 2*t43 + 2*t45 + 2*t46 &
      + 2*t47 + 2*t48)*t7(p)
   end do

   do p = 1, nc
      t60(p) = 2*t20*t40*cmp1(p)*t8(p) + (0.780549873862425d0*t12*t17*t20*t21*t29*t34 + &
      0.133921186137575d0*t12*t20*t21*t24*t29 + 2*t43 + 2*t45 + 2*t46 + &
      2*t47 + 2*t48)*t7(p)
   end do

   t61 = 2*R*t25*t31 + 4*t13*t20*t41

   t62 = t21*t27*t9

   t63 = -R*t11*t16*t25*t31 - R*t39 + 2*t13*t20*t40

   t64 = -2*R*T*t11*t16*t33*t42 - 2*R*T*t31*t35 - 0.323314343862425d0*t12*t17*t20 &
      *t21*t24*t9*(-0.82842712474619d0*t16*t18*t36 - 2.0d0*t16*t22) + &
      0.323314343862425d0*t12*t17*t20*t21*t28*t34*t9 + &
      0.323314343862425d0*t12*t17*t20*t24*t30*t37*t9 - &
      0.323314343862425d0*t12*t20*t21*t24*t28*t9

   t65 = t17*t20*t21*t24*t37*t9

   t66 = R*T*t23*t33*t38*t42 - 0.161657171931213d0*t12*t17*t20*t21*t24*t37 - &
      0.161657171931213d0*t12*t17*t20*t21*t24*t9*(0.82842712474619d0* &
      t11*t18*t23*t36 + 2.0d0*t11*t22*t23) + 0.161657171931213d0*t12* &
      t17*t20*t21*t29*t34*t9 - 0.161657171931213d0*t12*t20*t21*t24*t29* &
      t9

   res = -R*T*t39*t9 - 0.161657171931213d0*t12*t20*t21*t27*t9
   f = res

   res = -R*t39*t9 - 0.323314343862425d0*t13*t20*t21*t27*t9
   dfdT = res

   res = -R*T*t11*t31*t35 - 0.161657171931213d0*t12*t17*t20*t21*t24*t37*t9
   dfdV = res

   do i = 1, nc
      res = 2*t44*t7(i) - 0.323314343862425d0*t54*cmp1(i)*t8(i) + t55
      dfdn(i) = res
   end do

   res = -0.161657171931213d0*t21*t27*t9*(2*t14*t20 + 2*t15*t20)
   dfdT2 = res

   res = 2*R*T*t11*t31*t42 + R*T*t16*t33*t38/V**4 + 0.161657171931213d0*t12 &
         *t19*t20*t21*t24*t49*t9 + 0.161657171931213d0*t12*t20*t21*t34*t49 &
         *t9
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = 2*t44*mat2(i, j) - 0.323314343862425d0*t54*(1.0d0 - kij(i, j))* &
               cmp1(i)*cmp1(j) + t56*t7(i)*t7(j) + 4*t57*cmp1(i)*t7(j)*t8(i) + 4 &
               *t57*cmp1(j)*t7(i)*t8(j) + t58 + t59(i) + t60(j)
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t11*t31*t35 - 0.323314343862425d0*t13*t17*t20*t21*t24*t37*t9
   dfdTdV = res

   do i = 1, nc
      res = t61*t7(i) - 0.161657171931213d0*t62*(2*t20*cmp1(i)*t10(i) + 2*t20* &
            cmp1_T(i)*t8(i)) + t63
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = t64*t7(i) - 0.323314343862425d0*t65*cmp1(i)*t8(i) + t66
      dfdndV(i) = res
   end do

end subroutine pr76_c6m_w6
