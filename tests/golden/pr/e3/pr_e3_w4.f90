subroutine pr_e3_w4(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn)
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
   real(8) :: res, t11, t12, t13, t14, t15, t16, t6, t7, t8
   real(8) :: t1(nc), t10(nc), t2(nc), t9(nc)
   real(8) :: t3(nc, nc), t4(nc, nc), t5(nc, nc)
   integer :: i, l, m, p, q

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdn)) then
      t7 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t7 = t7 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdT)) then
      t8 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t8 = t8 + 2*n(l)*n(m)*t3(l, m)*t4(l, m)/T
         end do
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t9(i) = 0.0d0
         do l = 1, nc
            t9(i) = t9(i) + (1 - k(l, i))*n(l)*t3(i, l)
         end do
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t10(i) = 0.0d0
         do m = 1, nc
            t10(i) = t10(i) + n(m)*t3(i, m)*t4(i, m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t11 = V - t6
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t12 = V*(V + t6) + t11*t6
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdn)) then
      t13 = 1d0/t12
   end if

   if (present(dfdV) .or. present(dfdn)) then
      t14 = t12**(-2)
   end if

   if (present(dfdV) .or. present(dfdn)) then
      t15 = t11**(-2)
   end if

   if (present(f) .or. present(dfdT)) then
      t16 = 1d0/t11
   end if

   if (present(f)) then
      res = R*T*t16 - t13*t7
      f = res
   end if

   if (present(dfdT)) then
      res = R*t16 - t13*t8
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t15 - t14*t7*(-2*V - 2*t6)
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = R*T*t15*b(i) - t13*(t10(i) + t9(i)) - t14*t7*(-V*b(i) - t11*b(i) + &
               t6*b(i))
         dfdn(i) = res
      end do
   end if

end subroutine pr_e3_w4
