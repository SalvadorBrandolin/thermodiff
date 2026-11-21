subroutine pr76_a10_w3(nc, V, T, Pc, Tc, kij, lij, n, w, f, dfdn)
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
   real(8) :: res, t15, t16, t17, t19, t20, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34
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

   t19 = 0.0d0
   do k = 1, nc
      t19 = t19 + 0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t15
   end do

   t20 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t20 = t20 + n(k)*n(l)*t3(k, l)*t4(k, l)/t15
      end do
   end do

   do i = 1, nc
      t21(i) = 0.0d0
      do k = 1, nc
         t21(i) = t21(i) + n(k)*t3(i, k)*t4(i, k)/t15
      end do
   end do

   t22 = 0.0d0
   do k = 1, nc
      t22 = t22 + -0.07779607d0*R*Tc(k)*t1(k)*t5(k)*t6(k)/t15**2
   end do

   t23 = 0.0d0
   do k = 1, nc
      do l = k + 1, nc
         t23 = t23 + -n(k)*n(l)*t3(k, l)*t4(k, l)/t15**2
      end do
   end do

   t24 = t19 + 2*t20

   t25 = V - 0.414213562373095d0*t19 - 0.82842712474619d0*t20

   t26 = V + 2.41421356237309d0*t19 + 4.82842712474619d0*t20

   t27 = 1d0/t25

   t28 = 2*t23

   t29 = 1d0/V

   t30 = log(t26*t27)

   t31 = -t24*t29 + 1.0d0

   t32 = t16 + 2*t17

   t33 = 1d0/t24

   t34 = log(t31)

   res = -R*T*t15*t34 - 0.353553390593274d0*t30*t32*t33
   f = res

   do i = 1, nc
      res = R*T*t15*t29*(t22 + t28 + 2*t21(i))/t31 - R*T*t34 - &
            0.353553390593274d0*t25*t32*t33*(2.41421356237309d0*t27*(t22 + &
            t28 + 2*t21(i)) + 0.414213562373095d0*t26*(t22 + t28 + 2*t21(i))/ &
            t25**2)/t26 - 0.707106781186548d0*t30*t33*t18(i) + &
            0.353553390593274d0*t30*t32*(t22 + t28 + 2*t21(i))/t24**2
      dfdn(i) = res
   end do

end subroutine pr76_a10_w3
