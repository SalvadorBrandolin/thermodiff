subroutine pr_p33_w5a(nc, V, T, Tc, ac, b, k, n, f, dfdV, dfdV2)
   use yaeos__constants, only: R
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: V
   real(8), intent(in) :: T
   real(8), contiguous, intent(in) :: Tc(:)
   real(8), contiguous, intent(in) :: ac(:)
   real(8), contiguous, intent(in) :: b(:)
   real(8), contiguous, intent(in) :: k(:, :)
   real(8), contiguous, intent(in) :: n(:)
   real(8), optional, intent(out) :: f
   real(8), optional, intent(out) :: dfdV
   real(8), optional, intent(out) :: dfdV2
   real(8) :: res, t10, t11, t12, t13, t4, t6, t7, t8, t9
   real(8) :: t1(nc), t2(nc), t3(nc), t5(nc)
   integer :: l, m, p

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         t2(p) = sqrt(Tc(p)**(-2))
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do p = 1, nc
         t3(p) = sqrt(ac(p))
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      do l = 1, nc
         t5(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t5(l) = t5(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t6 = 0.0d0
      do l = 1, nc
         t6 = t6 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t7 = 0.0d0
      do l = 1, nc
         t7 = t7 + n(l)*t2(l)*t3(l)*t5(l)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t8 = V + t6
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdV2)) then
      t9 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdV)) then
      t10 = V - t6
   end if

   if (present(dfdV2)) then
      t11 = -V + t6
   end if

   if (present(f) .or. present(dfdV)) then
      t12 = V*t8 + t10*t6
   end if

   if (present(dfdV2)) then
      t13 = -V*t8 + t11*t6
   end if

   if (present(f)) then
      res = R*T/t10 - t7*t9/t12
      f = res
   end if

   if (present(dfdV)) then
      res = -R*T/t10**2 - t7*t9*(-2*V - 2*t6)/t12**2
      dfdV = res
   end if

   if (present(dfdV2)) then
      res = -2*R*T/t11**3 + 2*t7*t9/t13**2 + 8*t7*t8**2*t9/t13**3
      dfdV2 = res
   end if

end subroutine pr_p33_w5a
