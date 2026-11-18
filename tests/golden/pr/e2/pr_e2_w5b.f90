subroutine pr_e2_w5b(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdT2, dfdTdV)
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
   real(8) :: res, t10, t11, t12, t13, t14, t15, t16, t17, t7, t8, t9
   real(8) :: t1(nc), t2(nc)
   real(8) :: t3(nc, nc), t4(nc, nc), t5(nc, nc), t6(nc, nc)
   integer :: l, m, p, q

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(dfdT) .or. present(dfdTdV)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = 2*n(p)*n(q)*t3(p, q)*t4(p, q)/T
         end do
      end do
   end if

   if (present(f) .or. present(dfdV)) then
      do p = 1, nc
         do q = 1, nc
            t6(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t7 = 0.0d0
      do l = 1, nc
         t7 = t7 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdV)) then
      t8 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t8 = t8 + t6(l, m)
         end do
      end do
   end if

   if (present(dfdT) .or. present(dfdTdV)) then
      t9 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t9 = t9 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdT2)) then
      t10 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t10 = t10 + 2*n(l)*n(m)*t3(l, m)*t4(l, m)/T**2
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t11 = V - t7
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdT2) .or. present(dfdTdV)) then
      t12 = V*(V + t7) + t11*t7
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdT2)) then
      t13 = 1d0/t12
   end if

   if (present(dfdV) .or. present(dfdTdV)) then
      t14 = -2*V - 2*t7
   end if

   if (present(dfdV) .or. present(dfdTdV)) then
      t15 = t12**(-2)
   end if

   if (present(dfdV) .or. present(dfdTdV)) then
      t16 = t11**(-2)
   end if

   if (present(f) .or. present(dfdT)) then
      t17 = 1d0/t11
   end if

   if (present(f)) then
      res = R*T*t17 - t13*t8
      f = res
   end if

   if (present(dfdT)) then
      res = R*t17 - t13*t9
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t16 - t14*t15*t8
      dfdV = res
   end if

   if (present(dfdT2)) then
      res = -t10*t13
      dfdT2 = res
   end if

   if (present(dfdTdV)) then
      res = -R*t16 - t14*t15*t9
      dfdTdV = res
   end if

end subroutine pr_e2_w5b
