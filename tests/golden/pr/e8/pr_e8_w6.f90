subroutine pr_e8_w6(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), intent(in) :: Tc(:)
   real(8), intent(in) :: ac(:)
   real(8), intent(in) :: b(:)
   real(8), intent(in) :: k(:, :)
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f
   real(8), optional, intent(out) :: dfdT
   real(8), optional, intent(out) :: dfdV
   real(8), optional, intent(out) :: dfdn(:)
   real(8), optional, intent(out) :: dfdT2
   real(8), optional, intent(out) :: dfdV2
   real(8), optional, intent(out) :: dfdndn(:, :)
   real(8), optional, intent(out) :: dfdTdV
   real(8), optional, intent(out) :: dfdndT(:)
   real(8), optional, intent(out) :: dfdndV(:)
   real(8) :: res, t10, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31
   real(8) :: t32, t34, t35, t36, t37, t4, t5, t6, t7, t8
   real(8) :: t1(nc), t11(nc), t2(nc), t3(nc), t33(nc), t9(nc)
   integer :: i, j, l, m, p, q

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do m = 1, nc
         t9(m) = 0.0d0
         do l = 1, nc
            t9(m) = t9(m) + (1 - k(l, m))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t10 = 0.0d0
      do l = 1, nc
         t10 = t10 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT) .or. present(dfdndV)) then
      do i = 1, nc
         t11(i) = 0.0d0
         do m = 1, nc
            t11(i) = t11(i) + (1 - k(i, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t12 = 0.0d0
      do m = 1, nc
         t12 = t12 + n(m)*t2(m)*t3(m)*t9(m)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t13 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. &
        present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t14 = V - t10
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t15 = V + t10
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. &
        present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t16 = V*t15 + t10*t14
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. &
        present(dfdndV)) then
      t17 = t16**(-2)
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. &
        present(dfdndV)) then
      t18 = -2*V
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT) .or. present(dfdndV)) then
      t19 = 2*t10 + t18
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. present(dfdndT)) then
      t20 = 1d0/t16
   end if

   if (present(dfdT) .or. present(dfdTdV) .or. present(dfdndT)) then
      t21 = 1d0/T
   end if

   if (present(dfdV2) .or. present(dfdTdV)) then
      t22 = -V + t10
   end if

   if (present(dfdV) .or. present(dfdTdV) .or. present(dfdndV)) then
      t23 = -2*t10 + t18
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndT)) then
      t24 = t14**(-2)
   end if

   if (present(dfdV2)) then
      t25 = -V*t15 + t10*t22
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t26 = t16**(-3)
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t27 = t14**(-3)
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t28 = -4*V
   end if

   if (present(f) .or. present(dfdT)) then
      t29 = 1d0/t14
   end if

   if (present(dfdn)) then
      t30 = R*T*t24 - t12*t13*t17*t19
   end if

   if (present(dfdndn)) then
      t31 = 2*R*T*t27 - 2*t12*t13*t17 - t12*t13*t19*t26*(4*t10 + t28)
   end if

   if (present(dfdndn)) then
      t32 = t17*t19
   end if

   if (present(dfdndn)) then
      do q = 1, nc
         t33(q) = t13*t11(q)*t2(q)*t3(q) + t13*t2(q)*t3(q)*t9(q)
      end do
   end if

   if (present(dfdndT)) then
      t34 = R*t24 - 2*t12*t13*t17*t19*t21
   end if

   if (present(dfdndT)) then
      t35 = t13*t21
   end if

   if (present(dfdndV)) then
      t36 = -2*R*T*t27 + 2*t12*t13*t17 - t12*t13*t19*t26*(-4*t10 + t28)
   end if

   if (present(dfdndV)) then
      t37 = t17*t23
   end if

   if (present(f)) then
      res = R*T*t29 - t12*t13*t20
      f = res
   end if

   if (present(dfdT)) then
      res = R*t29 - 2*t12*t13*t20*t21
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t24 - t12*t13*t17*t23
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t20*(t13*t11(i)*t2(i)*t3(i) + t13*t2(i)*t3(i)*t9(i)) + t30*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdT2)) then
      res = -2*t12*t13*t20/T**2
      dfdT2 = res
   end if

   if (present(dfdV2)) then
      res = -2*R*T/t22**3 + 8*t12*t13*t15**2/t25**3 + 2*t12*t13/t25**2
      dfdV2 = res
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = i, nc
            res = -t20*(t13*(1 - k(i, j))*t2(i)*t2(j)*t3(i)*t3(j) + t13*(1 - k(j, i &
                  ))*t2(i)*t2(j)*t3(i)*t3(j)) + t31*b(i)*b(j) - t32*b(i)*t33(j) - &
                  t32*b(j)*t33(i)
            dfdndn(i, j) = res
            dfdndn(j, i) = res
         end do
      end do
   end if

   if (present(dfdTdV)) then
      res = -R/t22**2 - 2*t12*t13*t17*t21*t23
      dfdTdV = res
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = -t20*(2*t35*t11(i)*t2(i)*t3(i) + 2*t35*t2(i)*t3(i)*t9(i)) + t34*b( &
               i)
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = t36*b(i) - t37*(t13*t11(i)*t2(i)*t3(i) + t13*t2(i)*t3(i)*t9(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_e8_w6
