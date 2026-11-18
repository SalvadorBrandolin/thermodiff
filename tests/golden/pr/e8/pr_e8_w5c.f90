subroutine pr_e8_w5c(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndn)
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
   real(8), optional, intent(out) :: dfdn(:)
   real(8), optional, intent(out) :: dfdndn(:, :)
   real(8) :: res, t10, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t4, t5, t6, t7, t8
   real(8) :: t1(nc), t11(nc), t2(nc), t22(nc), t3(nc), t9(nc)
   integer :: i, j, l, m, p, q

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do m = 1, nc
         t9(m) = 0.0d0
         do l = 1, nc
            t9(m) = t9(m) + (1 - k(l, m))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t10 = 0.0d0
      do l = 1, nc
         t10 = t10 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do i = 1, nc
         t11(i) = 0.0d0
         do m = 1, nc
            t11(i) = t11(i) + (1 - k(i, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t12 = 0.0d0
      do m = 1, nc
         t12 = t12 + n(m)*t2(m)*t3(m)*t9(m)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t13 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t14 = V - t10
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t15 = V*(V + t10) + t10*t14
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t16 = -2*V + 2*t10
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t17 = t15**(-2)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t18 = 1d0/t15
   end if

   if (present(dfdn)) then
      t19 = R*T/t14**2 - t12*t13*t16*t17
   end if

   if (present(dfdndn)) then
      t20 = 2*R*T/t14**3 - 2*t12*t13*t17 - t12*t13*t16*(-4*V + 4*t10)/t15**3
   end if

   if (present(dfdndn)) then
      t21 = t16*t17
   end if

   if (present(dfdndn)) then
      do q = 1, nc
         t22(q) = t13*t11(q)*t2(q)*t3(q) + t13*t2(q)*t3(q)*t9(q)
      end do
   end if

   if (present(f)) then
      res = R*T/t14 - t12*t13*t18
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t18*(t13*t11(i)*t2(i)*t3(i) + t13*t2(i)*t3(i)*t9(i)) + t19*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = i, nc
            res = -t18*(t13*(1 - k(i, j))*t2(i)*t2(j)*t3(i)*t3(j) + t13*(1 - k(j, i &
                  ))*t2(i)*t2(j)*t3(i)*t3(j)) + t20*b(i)*b(j) - t21*b(i)*t22(j) - &
                  t21*b(j)*t22(i)
            dfdndn(i, j) = res
            dfdndn(j, i) = res
         end do
      end do
   end if

end subroutine pr_e8_w5c
