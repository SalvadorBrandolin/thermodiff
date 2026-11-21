subroutine pr_p22_w4(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn)
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
   real(8) :: res, t10, t11, t12, t13, t14, t15, t16, t17, t4, t6, t8, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t5(nc), t7(nc)
   integer :: i, l, m, p

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      do l = 1, nc
         t5(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t5(l) = t5(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         t7(i) = 0.0d0
         do l = 1, nc
            t7(i) = t7(i) + (1 - k(l, i))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t8 = 0.0d0
      do l = 1, nc
         t8 = t8 + n(l)*t2(l)*t3(l)*t5(l)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t9 = V - t6
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t10 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn)) then
      t11 = V*(V + t6) + t6*t9
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdn)) then
      t12 = 1d0/t11
   end if

   if (present(dfdV) .or. present(dfdn)) then
      t13 = t9**(-2)
   end if

   if (present(dfdV) .or. present(dfdn)) then
      t14 = t11**(-2)
   end if

   if (present(dfdV) .or. present(dfdn)) then
      t15 = -2*V
   end if

   if (present(f) .or. present(dfdT)) then
      t16 = 1d0/t9
   end if

   if (present(dfdn)) then
      t17 = R*T*t13 - t10*t14*t8*(t15 + 2*t6)
   end if

   if (present(f)) then
      res = R*T*t16 - t10*t12*t8
      f = res
   end if

   if (present(dfdT)) then
      res = R*t16 - 2*t10*t12*t8/T
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t13 - t10*t14*t8*(t15 - 2*t6)
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t12*(t10*t2(i)*t3(i)*t5(i) + t10*t2(i)*t3(i)*t7(i)) + t17*b(i)
         dfdn(i) = res
      end do
   end if

end subroutine pr_p22_w4
