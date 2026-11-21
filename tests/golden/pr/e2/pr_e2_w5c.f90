subroutine pr_e2_w5c(nc, V, T, Tc, ac, b, k, n, f, dfdn, dfdndn)
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
   real(8) :: res, t10, t11, t16, t17, t18, t19
   real(8) :: t1(nc), t12(nc), t13(nc), t14(nc), t15(nc), t2(nc), t6(nc)
   real(8) :: t3(nc, nc), t4(nc, nc), t5(nc, nc), t7(nc, nc), t8(nc, nc), t9(nc, nc)
   integer :: i, j, l, m, p, q

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t1(p) = b(p)*n(p)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         t6(p) = -V*b(p)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t7(p, q) = 1 - k(q, p)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t8(p, q) = n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do p = 1, nc
         do q = 1, nc
            t9(p, q) = n(q)*t3(p, q)*t7(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t10 = 0.0d0
      do l = 1, nc
         t10 = t10 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t11 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t11 = t11 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do i = 1, nc
         t12(i) = 0.0d0
         do m = 1, nc
            t12(i) = t12(i) + t8(i, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      do i = 1, nc
         t13(i) = 0.0d0
         do l = 1, nc
            t13(i) = t13(i) + t9(i, l)
         end do
      end do
   end if

   if (present(dfdndn)) then
      do j = 1, nc
         t14(j) = 0.0d0
         do m = 1, nc
            t14(j) = t14(j) + t8(j, m)
         end do
      end do
   end if

   if (present(dfdndn)) then
      do j = 1, nc
         t15(j) = 0.0d0
         do l = 1, nc
            t15(j) = t15(j) + t9(j, l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t16 = V - t10
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t17 = V*(V + t10) + t10*t16
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t18 = t17**(-2)
   end if

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t19 = 1d0/t17
   end if

   if (present(f)) then
      res = R*T/t16 - t11*t19
      f = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = R*T*b(i)/t16**2 - t11*t18*(t10*b(i) - t16*b(i) + t6(i)) - t19*(t12 &
               (i) + t13(i))
         dfdn(i) = res
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
            res = 2*R*T*b(i)*b(j)/t16**3 - 2*t11*t18*b(i)*b(j) - t11*(-2*V*b(j) + 2* &
                  t10*b(j) - 2*t16*b(j))*(t10*b(i) - t16*b(i) + t6(i))/t17**3 - t18 &
                  *(t12(i) + t13(i))*(t10*b(j) - t16*b(j) + t6(j)) - t18*(t14(j) + &
                  t15(j))*(t10*b(i) - t16*b(i) + t6(i)) - t19*(t3(i, j)*t4(i, j) + &
                  t3(i, j)*t7(i, j))
            dfdndn(i, j) = res
         end do
      end do
   end if

end subroutine pr_e2_w5c
