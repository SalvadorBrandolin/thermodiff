subroutine pr_e3_w5d(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndT, dfdndV)
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
   real(8), optional, intent(out) :: dfdndT(:)
   real(8), optional, intent(out) :: dfdndV(:)
   real(8) :: res, t10, t11, t16, t17, t18, t19, t20, t21, t22
   real(8) :: t1(nc), t12(nc), t13(nc), t14(nc), t15(nc), t2(nc), t7(nc)
   real(8) :: t3(nc, nc), t4(nc, nc), t5(nc, nc), t6(nc, nc), t8(nc, nc), t9(nc, nc)
   integer :: i, l, m, p, q

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t6(p, q) = 1 - k(q, p)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndT)) then
      do p = 1, nc
         t7(p) = -V*b(p)
      end do
   end if

   if (present(dfdn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t8(p, q) = n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t9(p, q) = n(q)*t3(p, q)*t6(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t10 = 0.0d0
      do l = 1, nc
         t10 = t10 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndV)) then
      t11 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t11 = t11 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndV)) then
      do i = 1, nc
         t12(i) = 0.0d0
         do m = 1, nc
            t12(i) = t12(i) + t8(i, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndV)) then
      do i = 1, nc
         t13(i) = 0.0d0
         do l = 1, nc
            t13(i) = t13(i) + t9(i, l)
         end do
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         t14(i) = 0.0d0
         do l = 1, nc
            t14(i) = t14(i) + 2*n(l)*t3(i, l)*t6(i, l)/T
         end do
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         t15(i) = 0.0d0
         do m = 1, nc
            t15(i) = t15(i) + 2*n(m)*t3(i, m)*t4(i, m)/T
         end do
      end do
   end if

   if (present(dfdndT)) then
      t16 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t16 = t16 + 2*n(l)*n(m)*t3(l, m)*t4(l, m)/T
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t17 = V - t10
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t18 = V*(V + t10) + t10*t17
   end if

   if (present(dfdn) .or. present(dfdndT) .or. present(dfdndV)) then
      t19 = t18**(-2)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndT)) then
      t20 = 1d0/t18
   end if

   if (present(dfdndV)) then
      t21 = -2*V - 2*t10
   end if

   if (present(dfdn) .or. present(dfdndT)) then
      t22 = t17**(-2)
   end if

   if (present(f)) then
      res = R*T/t17 - t11*t20
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = R*T*t22*b(i) - t11*t19*(t10*b(i) - t17*b(i) + t7(i)) - t20*(t12(i &
               ) + t13(i))
         dfdn(i) = res
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = R*t22*b(i) - t16*t19*(t10*b(i) - t17*b(i) + t7(i)) - t20*(t14(i) + &
               t15(i))
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = -2*R*T*b(i)/t17**3 + 2*t11*t19*b(i) - t11*t21*(-2*V*b(i) + 2*t10*b &
               (i) - 2*t17*b(i))/t18**3 - t19*t21*(t12(i) + t13(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_e3_w5d
