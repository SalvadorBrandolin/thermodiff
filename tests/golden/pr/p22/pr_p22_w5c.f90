subroutine pr_p22_w5c(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndn)
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
   real(8) :: res, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t4, t5, t6, t7, t9
   real(8) :: t1(nc), t10(nc), t2(nc), t21(nc), t3(nc), t8(nc)
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
      do l = 1, nc
         t8(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t8(l) = t8(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t9 = 0.0d0
      do l = 1, nc
         t9 = t9 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do i = 1, nc
         t10(i) = 0.0d0
         do l = 1, nc
            t10(i) = t10(i) + (1 - k(l, i))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t11 = 0.0d0
      do l = 1, nc
         t11 = t11 + n(l)*t2(l)*t3(l)*t8(l)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t12 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t13 = V - t9
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t14 = V*(V + t9) + t13*t9
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t15 = -2*V + 2*t9
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t16 = t14**(-2)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t17 = 1d0/t14
   end if

   if (present(dfdn)) then
      t18 = R*T/t13**2 - t11*t12*t15*t16
   end if

   if (present(dfdndn)) then
      t19 = 2*R*T/t13**3 - 2*t11*t12*t16 - t11*t12*t15*(-4*V + 4*t9)/t14**3
   end if

   if (present(dfdndn)) then
      t20 = t15*t16
   end if

   if (present(dfdndn)) then
      do q = 1, nc
         t21(q) = t12*t10(q)*t2(q)*t3(q) + t12*t2(q)*t3(q)*t8(q)
      end do
   end if

   if (present(f)) then
      res = R*T/t13 - t11*t12*t17
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t17*(t12*t10(i)*t2(i)*t3(i) + t12*t2(i)*t3(i)*t8(i)) + t18*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = i, nc
            res = -t17*(t12*(1 - k(i, j))*t2(i)*t2(j)*t3(i)*t3(j) + t12*(1 - k(j, i &
                  ))*t2(i)*t2(j)*t3(i)*t3(j)) + t19*b(i)*b(j) - t20*b(i)*t21(j) - &
                  t20*b(j)*t21(i)
            dfdndn(i, j) = res
            dfdndn(j, i) = res
         end do
      end do
   end if

end subroutine pr_p22_w5c
