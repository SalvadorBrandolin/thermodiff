subroutine pr_p32_w6(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
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
   real(8), optional, intent(out) :: dfdT
   real(8), optional, intent(out) :: dfdV
   real(8), contiguous, optional, intent(out) :: dfdn(:)
   real(8), optional, intent(out) :: dfdT2
   real(8), optional, intent(out) :: dfdV2
   real(8), contiguous, optional, intent(out) :: dfdndn(:, :)
   real(8), optional, intent(out) :: dfdTdV
   real(8), contiguous, optional, intent(out) :: dfdndT(:)
   real(8), contiguous, optional, intent(out) :: dfdndV(:)
   real(8) :: res, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31
   real(8) :: t33, t34, t35, t36, t4, t5, t6, t7, t9
   real(8) :: t1(nc), t10(nc), t2(nc), t3(nc), t32(nc), t8(nc)
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
      do l = 1, nc
         t8(l) = 0.0d0
      end do
      do m = 1, nc
         do l = 1, nc
            t8(l) = t8(l) + (1 - k(l, m))*n(m)*t2(m)*t3(m)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t9 = 0.0d0
      do l = 1, nc
         t9 = t9 + t1(l)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT) .or. present(dfdndV)) then
      do i = 1, nc
         t10(i) = 0.0d0
         do l = 1, nc
            t10(i) = t10(i) + (1 - k(l, i))*n(l)*t2(l)*t3(l)
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t11 = 0.0d0
      do l = 1, nc
         t11 = t11 + n(l)*t2(l)*t3(l)*t8(l)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t12 = sqrt(T**4)
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. &
        present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t13 = V - t9
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t14 = V + t9
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. &
        present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t15 = V*t14 + t13*t9
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. &
        present(dfdndV)) then
      t16 = t15**(-2)
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. &
        present(dfdndV)) then
      t17 = -2*V
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT) .or. present(dfdndV)) then
      t18 = t17 + 2*t9
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. present(dfdndT)) then
      t19 = 1d0/t15
   end if

   if (present(dfdT) .or. present(dfdTdV) .or. present(dfdndT)) then
      t20 = 1d0/T
   end if

   if (present(dfdV2) .or. present(dfdTdV)) then
      t21 = -V + t9
   end if

   if (present(dfdV) .or. present(dfdTdV) .or. present(dfdndV)) then
      t22 = t17 - 2*t9
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdndT)) then
      t23 = t13**(-2)
   end if

   if (present(dfdV2)) then
      t24 = -V*t14 + t21*t9
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t25 = t15**(-3)
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t26 = t13**(-3)
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      t27 = -4*V
   end if

   if (present(f) .or. present(dfdT)) then
      t28 = 1d0/t13
   end if

   if (present(dfdn)) then
      t29 = R*T*t23 - t11*t12*t16*t18
   end if

   if (present(dfdndn)) then
      t30 = 2*R*T*t26 - 2*t11*t12*t16 - t11*t12*t18*t25*(t27 + 4*t9)
   end if

   if (present(dfdndn)) then
      t31 = t16*t18
   end if

   if (present(dfdndn)) then
      do q = 1, nc
         t32(q) = t12*t10(q)*t2(q)*t3(q) + t12*t2(q)*t3(q)*t8(q)
      end do
   end if

   if (present(dfdndT)) then
      t33 = R*t23 - 2*t11*t12*t16*t18*t20
   end if

   if (present(dfdndT)) then
      t34 = t12*t20
   end if

   if (present(dfdndV)) then
      t35 = -2*R*T*t26 + 2*t11*t12*t16 - t11*t12*t18*t25*(t27 - 4*t9)
   end if

   if (present(dfdndV)) then
      t36 = t16*t22
   end if

   if (present(f)) then
      res = R*T*t28 - t11*t12*t19
      f = res
   end if

   if (present(dfdT)) then
      res = R*t28 - 2*t11*t12*t19*t20
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t23 - t11*t12*t16*t22
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = -t19*(t12*t10(i)*t2(i)*t3(i) + t12*t2(i)*t3(i)*t8(i)) + t29*b(i)
         dfdn(i) = res
      end do
   end if

   if (present(dfdT2)) then
      res = -2*t11*t12*t19/T**2
      dfdT2 = res
   end if

   if (present(dfdV2)) then
      res = -2*R*T/t21**3 + 8*t11*t12*t14**2/t24**3 + 2*t11*t12/t24**2
      dfdV2 = res
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = i, nc
            res = -t19*(t12*(1 - k(i, j))*t2(i)*t2(j)*t3(i)*t3(j) + t12*(1 - k(j, i &
                  ))*t2(i)*t2(j)*t3(i)*t3(j)) + t30*b(i)*b(j) - t31*b(i)*t32(j) - &
                  t31*b(j)*t32(i)
            dfdndn(i, j) = res
            dfdndn(j, i) = res
         end do
      end do
   end if

   if (present(dfdTdV)) then
      res = -R/t21**2 - 2*t11*t12*t16*t20*t22
      dfdTdV = res
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = -t19*(2*t34*t10(i)*t2(i)*t3(i) + 2*t34*t2(i)*t3(i)*t8(i)) + t33*b( &
               i)
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = t35*b(i) - t36*(t12*t10(i)*t2(i)*t3(i) + t12*t2(i)*t3(i)*t8(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_p32_w6
