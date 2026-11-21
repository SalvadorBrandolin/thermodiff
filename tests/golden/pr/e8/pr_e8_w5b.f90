subroutine pr_e8_w5b(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdT2, dfdTdV)
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
   real(8), optional, intent(out) :: dfdT2
   real(8), optional, intent(out) :: dfdTdV
   real(8) :: res, t10, t11, t12, t13, t14, t15, t4, t6, t7, t8, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t5(nc)
   integer :: l, m, p

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do m = 1, nc
         t5(m) = 0.0d0
         do l = 1, nc
            t5(m) = t5(m) + (1 - k(l, m))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t7 = 0.0d0
      do m = 1, nc
         t7 = t7 + n(m)*t2(m)*t3(m)*t5(m)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t8 = V - t6
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t9 = V*(V + t6) + t6*t8
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t10 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdT2)) then
      t11 = 1d0/t9
   end if

   if (present(dfdV) .or. present(dfdTdV)) then
      t12 = -2*V - 2*t6
   end if

   if (present(dfdV) .or. present(dfdTdV)) then
      t13 = t9**(-2)
   end if

   if (present(f) .or. present(dfdT)) then
      t14 = 1d0/t8
   end if

   if (present(dfdT) .or. present(dfdTdV)) then
      t15 = 1d0/T
   end if

   if (present(f)) then
      res = R*T*t14 - t10*t11*t7
      f = res
   end if

   if (present(dfdT)) then
      res = R*t14 - 2*t10*t11*t15*t7
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T/t8**2 - t10*t12*t13*t7
      dfdV = res
   end if

   if (present(dfdT2)) then
      res = -2*t10*t11*t7/T**2
      dfdT2 = res
   end if

   if (present(dfdTdV)) then
      res = -R/(-V + t6)**2 - 2*t10*t12*t13*t15*t7
      dfdTdV = res
   end if

end subroutine pr_e8_w5b
