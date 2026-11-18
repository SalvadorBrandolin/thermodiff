subroutine pr76_c6n_w6(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, &
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
   real(8) :: res, t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t2, t20, t21, t22, t23, t24, t25, t26, t27, t28
   real(8) :: t29, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t4, t40, t41, t42, t43, t44, t45, t46, t47, t48
   real(8) :: t49, t5, t50, t51, t53, t54, t55, t56, t59, t60, t61, t62, t63, t8
   real(8) :: cmp0(nc), cmp0_T(nc), cmp0_TT(nc), mat1_g(nc), mat1_gT(nc), t52(nc), t57(nc), t58(nc), t6(nc), t7(nc), t9(nc)
   real(8) :: mat1(nc, nc), mat2(nc, nc)
   integer :: i, ia, ib, ic, j, k, l, m, p

   do ic = 1, nc
      cmp0(ic) = ((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0 &
      ) + 1)**2*Tc(ic)**2/Pc(ic)
   end do

   t49 = 1d0/T

   do ic = 1, nc
      cmp0_T(ic) = -t49*sqrt(T/Tc(ic))*((1 - sqrt(T/Tc(ic)))*(-0.26993d0*w(ic)**2 + &
      1.54226d0*w(ic) + 0.37464d0) + 1)*(-0.26993d0*w(ic)**2 + &
      1.54226d0*w(ic) + 0.37464d0)*Tc(ic)**2/Pc(ic)
   end do

   t50 = T**(-2)

   do ic = 1, nc
      cmp0_TT(ic) = (1.0d0/2.0d0)*t49*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0)**2 &
      *Tc(ic)/Pc(ic) + (1.0d0/2.0d0)*t50*sqrt(T/Tc(ic))*((1 - sqrt(T/Tc &
      (ic)))*(-0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0) + 1)*( &
      -0.26993d0*w(ic)**2 + 1.54226d0*w(ic) + 0.37464d0)*Tc(ic)**2/Pc( &
      ic)
   end do

   do ic = 1, nc
      mat1_g(ic) = (1.0d0/2.0d0)*cmp0_T(ic)/cmp0(ic)
   end do

   do ic = 1, nc
      mat1_gT(ic) = (1.0d0/2.0d0)*cmp0_TT(ic)/cmp0(ic) - 1.0d0/2.0d0*cmp0_T(ic)**2/cmp0(ic) &
      **2
   end do

   t51 = R**4

   do ia = 1, nc
      do ib = ia, nc
         mat1(ia, ib) = sqrt(t51*cmp0(ia)*cmp0(ib))*(1.0d0 - kij(ia, ib))
         mat1(ib, ia) = mat1(ia, ib)
      end do
   end do

   do p = 1, nc
      t52(p) = 0.038898035d0*R*Tc(p)/Pc(p)
   end do

   do ia = 1, nc
      do ib = ia, nc
         mat2(ia, ib) = (1.0d0 - lij(ia, ib))*(t52(ia) + t52(ib))
         mat2(ib, ia) = mat2(ia, ib)
      end do
   end do

   do l = 1, nc
      t6(l) = 0.0d0
      do k = 1, nc
         t6(l) = t6(l) + mat1(k, l)*n(k)
      end do
   end do

   do l = 1, nc
      t7(l) = 0.0d0
      do k = 1, nc
         t7(l) = t7(l) + mat2(k, l)*n(k)
      end do
   end do

   t8 = 0.0d0
   do m = 1, nc
      t8 = t8 + n(m)
   end do

   do l = 1, nc
      t9(l) = 0.0d0
      do k = 1, nc
         t9(l) = t9(l) + (mat1_g(k) + mat1_g(l))*mat1(k, l)*n(k)
      end do
   end do

   t10 = 0.0d0
   do k = 1, nc
      t10 = t10 + (2*mat1(k, k)*mat1_gT(k) + 4*mat1(k, k)*mat1_g(k)**2)*n(k)**2
   end do

   t11 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t11 = t11 + ((mat1_gT(k) + mat1_gT(l))*mat1(k, l) + (mat1_g(k) + mat1_g(l))**2*mat1( &
      k, l))*n(k)*n(l)
      end do
   end do

   t12 = 0.0d0
   do l = 1, nc
      t12 = t12 + n(l)*t6(l)
   end do

   t13 = 0.0d0
   do l = 1, nc
      t13 = t13 + n(l)*t7(l)
   end do

   t14 = 0.0d0
   do l = 1, nc
      t14 = t14 + n(l)*t9(l)
   end do

   t15 = 1d0/t8

   t16 = V - 0.414213562373095d0*t13*t15

   t17 = V + 2.41421356237309d0*t13*t15

   t18 = 1d0/t16

   t19 = 1d0/t13

   t20 = t16**(-2)

   t21 = t8**(-2)

   t22 = 1d0/t17

   t23 = 1d0/V

   t24 = -t13*t15*t23 + 1.0d0

   t25 = log(t17*t18)

   t26 = 0.414213562373095d0*t15*t17*t20 + 2.41421356237309d0*t15*t18

   t27 = -0.414213562373095d0*t13*t17*t20*t21 - 2.41421356237309d0*t13*t18*t21

   t28 = t13**(-2)

   t29 = 1d0/t24

   t30 = t8**(-3)

   t31 = t24**(-2)

   t32 = t17**(-2)

   t33 = V**(-2)

   t34 = t16**(-3)

   t35 = -t17*t20 + t18

   t36 = t13**2

   t37 = log(t24)

   t38 = -0.161657171931213d0*t16*t19*t22*t27*t8 - 0.161657171931213d0*t19*t25

   t39 = -0.161657171931213d0*t16*t19*t22*t26*t8 + 0.161657171931213d0*t25*t28*t8

   t40 = V**(-3)

   t41 = -0.161657171931213d0*t12*t16*t19*t22*t8*(-0.34314575050762d0*t13*t17*t30 &
      *t34 - 2.0d0*t13*t20*t30 - 0.414213562373095d0*t17*t20*t21 - &
      2.41421356237309d0*t18*t21)

   t42 = R*T*t23*t29 - 0.161657171931213d0*t12*t16*t19*t22*t26*t8 + &
      0.161657171931213d0*t12*t25*t28*t8

   t43 = 0.161657171931213d0*t12*t16*t22*t27*t28*t8

   t44 = -R*T*t13*t21*t31*t33

   t45 = -0.161657171931213d0*t12*t16*t19*t22*t26

   t46 = 0.161657171931213d0*t12*t25*t28

   t47 = -t17*t18 + 1

   t48 = t8**(-4)

   t53 = t19*t25*t8

   t54 = -R*T*t13*t15*t23*t29 - R*T*t37 - 0.161657171931213d0*t12*t16*t19*t22*t27 &
      *t8 - 0.161657171931213d0*t12*t19*t25

   t55 = 4*R*T*t15*t31*t33 - 0.64662868772485d0*t12*t16*t19*t22*t8*( &
      0.34314575050762d0*t17*t21*t34 + 2.0d0*t20*t21) + &
      1.56109974772485d0*t12*t16*t19*t26*t32 + 1.2932573754497d0*t12* &
      t16*t22*t26*t28*t8 + 0.26784237227515d0*t12*t19*t22*t26 - &
      1.2932573754497d0*t12*t25*t8/t13**3

   t56 = R*T*t30*t31*t33*t36 - 0.390274936931213d0*t12*t15*t16*t27*t32 - &
      0.0669605930687875d0*t12*t15*t22*t27 - 0.323314343862425d0*t12* &
      t16*t19*t22*t27 - 0.161657171931213d0*t12*t16*t19*t22*t8*( &
      0.82842712474619d0*t13*t17*t20*t30 + 4.82842712474619d0*t13*t18* &
      t30 + 0.34314575050762d0*t17*t34*t36*t48 + 2.0d0*t20*t36*t48)

   do p = 1, nc
      t57(p) = 2*t38*t6(p) + (-0.780549873862425d0*t12*t15*t16*t26*t32 - &
      0.133921186137575d0*t12*t15*t22*t26 + 2*t41 + 2*t43 + 2*t44 + 2* &
      t45 + 2*t46)*t7(p)
   end do

   do p = 1, nc
      t58(p) = 2*t38*t6(p) + (0.780549873862425d0*t12*t16*t19*t27*t32 + &
      0.133921186137575d0*t12*t19*t22*t27 + 2*t41 + 2*t43 + 2*t44 + 2* &
      t45 + 2*t46)*t7(p)
   end do

   t59 = 2*R*t23*t29 + 2*t14*t39

   t60 = -R*t13*t15*t23*t29 - R*t37 + t14*t38

   t61 = -2*R*T*t13*t15*t31*t40 - 2*R*T*t29*t33 - 0.323314343862425d0*t12*t16*t19 &
      *t22*t8*(-0.82842712474619d0*t15*t17*t34 - 2.0d0*t15*t20) + &
      0.323314343862425d0*t12*t16*t19*t26*t32*t8 + 0.323314343862425d0* &
      t12*t16*t22*t28*t35*t8 - 0.323314343862425d0*t12*t19*t22*t26*t8

   t62 = t16*t19*t22*t35*t8

   t63 = R*T*t21*t31*t36*t40 - 0.161657171931213d0*t12*t16*t19*t22*t35 - &
      0.161657171931213d0*t12*t16*t19*t22*t8*(0.82842712474619d0*t13* &
      t17*t21*t34 + 2.0d0*t13*t20*t21) + 0.161657171931213d0*t12*t16* &
      t19*t27*t32*t8 - 0.161657171931213d0*t12*t19*t22*t27*t8

   res = -R*T*t37*t8 - 0.161657171931213d0*t12*t19*t25*t8
   f = res

   res = -R*t37*t8 - 0.161657171931213d0*t14*t19*t25*t8
   dfdT = res

   res = -R*T*t13*t29*t33 - 0.161657171931213d0*t12*t16*t19*t22*t35*t8
   dfdV = res

   do i = 1, nc
      res = 2*t42*t7(i) - 0.323314343862425d0*t53*t6(i) + t54
      dfdn(i) = res
   end do

   res = -0.161657171931213d0*t19*t25*t8*(t10 + 2*t11)
   dfdT2 = res

   res = 2*R*T*t13*t29*t40 + R*T*t15*t31*t36/V**4 + 0.161657171931213d0*t12 &
         *t18*t19*t22*t47*t8 + 0.161657171931213d0*t12*t19*t32*t47*t8
   dfdV2 = res

   do i = 1, nc
      do j = i, nc
         res = 4*t39*t6(i)*t7(j) + 4*t39*t6(j)*t7(i) + 2*t42*mat2(i, j) - &
               0.323314343862425d0*t53*mat1(i, j) + t55*t7(i)*t7(j) + t56 + t57( &
               i) + t58(j)
         dfdndn(i, j) = res
         dfdndn(j, i) = res
      end do
   end do

   res = -R*t13*t29*t33 - 0.161657171931213d0*t14*t16*t19*t22*t35*t8
   dfdTdV = res

   do i = 1, nc
      res = -0.323314343862425d0*t53*t9(i) + t59*t7(i) + t60
      dfdndT(i) = res
   end do

   do i = 1, nc
      res = t61*t7(i) - 0.323314343862425d0*t62*t6(i) + t63
      dfdndV(i) = res
   end do

end subroutine pr76_c6n_w6
