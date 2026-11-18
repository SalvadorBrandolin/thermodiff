subroutine pr76_a10h_w3(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdn)
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
   real(8), intent(out) :: dfdn(:)
   real(8) :: res, t15, t16, t17, t19, t20, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37
   real(8) :: t38
   real(8) :: t1(nc), t12(nc), t18(nc), t2(nc), t21(nc), t5(nc), t6(nc), t7(nc), t8(nc), t9(nc)
   real(8) :: t10(nc, nc), t11(nc, nc), t13(nc, nc), t14(nc, nc), t3(nc, nc), t4(nc, nc)
   integer :: i, k, l, m, p, q

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
      t7(p) = (1 - sqrt(T/Tc(p)))*(-0.26993d0*w(p)**2 + 1.54226d0*w(p) + 0.37464d0) + &
      1
   end do

   do p = 1, nc
      t8(p) = Tc(p)**2
   end do

   do p = 1, nc
      t9(p) = t7(p)**2
   end do

   do p = 1, nc
      do q = p, nc
         t10(p, q) = sqrt(R**4*t1(p)*t1(q)*t8(p)*t8(q)*t9(p)*t9(q))
         t10(q, p) = t10(p, q)
      end do
   end do

   do p = 1, nc
      do q = p, nc
         t11(p, q) = 1.0d0 - kij(p, q)
         t11(q, p) = t11(p, q)
      end do
   end do

   do p = 1, nc
      t12(p) = 0.45723553d0*sqrt(R**4*Tc(p)**4*t7(p)**4/Pc(p)**2)*(1.0d0 - kij(p, p))* &
      t5(p)
   end do

   do p = 1, nc
      do q = 1, nc
         t13(p, q) = 0.45723553d0*n(p)*n(q)*t10(p, q)*t11(p, q)
      end do
   end do

   do p = 1, nc
      do q = 1, nc
         t14(p, q) = 0.45723553d0*n(q)*t10(p, q)*t11(p, q)
      end do
   end do

   t15 = 0.0d0
   do m = 1, nc
      t15 = t15 + n(m)
   end do

   t16 = 0.0d0
   do k = 1, nc
      t16 = t16 + t12(k)
   end do

   t17 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t17 = t17 + t13(k, l)
      end do
   end do

   do i = 1, nc
      t18(i) = 0.0d0
      do k = 1, nc
         t18(i) = t18(i) + t14(i, k)
      end do
   end do

   t24 = R/t15

   t19 = 0.0d0
   do k = 1, nc
      t19 = t19 + 0.07779607d0*t24*Tc(k)*t1(k)*t5(k)*t6(k)
   end do

   t25 = 1d0/t15

   t20 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t20 = t20 + t25*n(k)*n(l)*t3(k, l)*t4(k, l)
      end do
   end do

   do i = 1, nc
      t21(i) = 0.0d0
      do k = 1, nc
         t21(i) = t21(i) + t25*n(k)*t3(i, k)*t4(i, k)
      end do
   end do

   t26 = R/t15**2

   t22 = 0.0d0
   do k = 1, nc
      t22 = t22 + -0.07779607d0*t26*Tc(k)*t1(k)*t5(k)*t6(k)
   end do

   t27 = t15**(-2)

   t23 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t23 = t23 + -t27*n(k)*n(l)*t3(k, l)*t4(k, l)
      end do
   end do

   t28 = t19 + 2*t20

   t29 = V - 0.414213562373095d0*t19 - 0.82842712474619d0*t20

   t30 = V + 2.41421356237309d0*t19 + 4.82842712474619d0*t20

   t31 = 1d0/t29

   t32 = 2*t23

   t33 = 1d0/V

   t34 = log(t30*t31)

   t35 = -t28*t33 + 1.0d0

   t36 = t16 + 2*t17

   t37 = 1d0/t28

   t38 = log(t35)

   res = -R*T*t15*t38 - 0.353553390593274d0*t34*t36*t37
   f = res

   do i = 1, nc
      res = R*T*t15*t33*(t22 + t32 + 2*t21(i))/t35 - R*T*t38 - &
            0.353553390593274d0*t29*t36*t37*(2.41421356237309d0*t31*(t22 + &
            t32 + 2*t21(i)) + 0.414213562373095d0*t30*(t22 + t32 + 2*t21(i))/ &
            t29**2)/t30 - 0.707106781186548d0*t34*t37*t18(i) + &
            0.353553390593274d0*t34*t36*(t22 + t32 + 2*t21(i))/t28**2
      dfdn(i) = res
   end do

end subroutine pr76_a10h_w3
