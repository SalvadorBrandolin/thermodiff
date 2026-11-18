subroutine pr_e2_w6(nc, V, T, Tc, ac, b, k, n, f, dfdT, dfdV, dfdn, dfdT2, dfdV2, dfdndn, dfdTdV, dfdndT, dfdndV)
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
   real(8), optional, intent(out) :: dfdT2
   real(8), optional, intent(out) :: dfdV2
   real(8), optional, intent(out) :: dfdndn(:, :)
   real(8), optional, intent(out) :: dfdTdV
   real(8), optional, intent(out) :: dfdndT(:)
   real(8), optional, intent(out) :: dfdndV(:)
   real(8) :: res, t12, t13, t14, t17, t22, t23, t24, t25, t26, t27, t28, t29, t30
   real(8) :: t1(nc), t11(nc), t15(nc), t16(nc), t18(nc), t19(nc), t2(nc), t20(nc), t21(nc), t7(nc)
   real(8) :: t10(nc, nc), t3(nc, nc), t4(nc, nc), t5(nc, nc), t6(nc, nc), t8(nc, nc), t9(nc, nc)
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
         t2(p) = Tc(p)**(-2)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t3(p, q) = sqrt(T**4*ac(p)*ac(q)*t2(p)*t2(q))
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t4(p, q) = 1 - k(p, q)
         end do
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdn) .or. present(dfdV2) .or. present(dfdndn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t5(p, q) = n(p)*n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t6(p, q) = 1 - k(q, p)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndT)) then
      do p = 1, nc
         t7(p) = -V*b(p)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t8(p, q) = n(q)*t3(p, q)*t4(p, q)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndV)) then
      do p = 1, nc
         do q = 1, nc
            t9(p, q) = n(q)*t3(p, q)*t6(p, q)
         end do
      end do
   end if

   if (present(dfdT) .or. present(dfdTdV) .or. present(dfdndT)) then
      do p = 1, nc
         do q = 1, nc
            t10(p, q) = 2*n(p)*n(q)*t3(p, q)*t4(p, q)/T
         end do
      end do
   end if

   if (present(dfdndn) .or. present(dfdndV)) then
      do p = 1, nc
         t11(p) = -2*V*b(p)
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t12 = 0.0d0
      do l = 1, nc
         t12 = t12 + t1(l)
      end do
   end if

   if (present(f) .or. present(dfdV) .or. present(dfdn) .or. present(dfdV2) .or. present(dfdndn) .or. present(dfdndV)) then
      t13 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t13 = t13 + t5(l, m)
         end do
      end do
   end if

   if (present(dfdT) .or. present(dfdTdV) .or. present(dfdndT)) then
      t14 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t14 = t14 + t10(l, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndV)) then
      do i = 1, nc
         t15(i) = 0.0d0
         do m = 1, nc
            t15(i) = t15(i) + t8(i, m)
         end do
      end do
   end if

   if (present(dfdn) .or. present(dfdndn) .or. present(dfdndV)) then
      do i = 1, nc
         t16(i) = 0.0d0
         do l = 1, nc
            t16(i) = t16(i) + t9(i, l)
         end do
      end do
   end if

   if (present(dfdT2)) then
      t17 = 0.0d0
      do m = 1, nc
         do l = 1, nc
            t17 = t17 + 2*n(l)*n(m)*t3(l, m)*t4(l, m)/T**2
         end do
      end do
   end if

   if (present(dfdndn)) then
      do j = 1, nc
         t18(j) = 0.0d0
         do m = 1, nc
            t18(j) = t18(j) + t8(j, m)
         end do
      end do
   end if

   if (present(dfdndn)) then
      do j = 1, nc
         t19(j) = 0.0d0
         do l = 1, nc
            t19(j) = t19(j) + t9(j, l)
         end do
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         t20(i) = 0.0d0
         do l = 1, nc
            t20(i) = t20(i) + 2*n(l)*t3(i, l)*t6(i, l)/T
         end do
      end do
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         t21(i) = 0.0d0
         do m = 1, nc
            t21(i) = t21(i) + 2*n(m)*t3(i, m)*t4(i, m)/T
         end do
      end do
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t22 = V - t12
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdV) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdV2) .or. &
        present(dfdndn) .or. present(dfdTdV) .or. present(dfdndT) .or. present(dfdndV)) then
      t23 = V*(V + t12) + t12*t22
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdV2) .or. present(dfdndn) .or. present(dfdTdV) .or. &
        present(dfdndT) .or. present(dfdndV)) then
      t24 = t23**(-2)
   end if

   if (present(f) .or. present(dfdT) .or. present(dfdn) .or. present(dfdT2) .or. present(dfdndn) .or. present(dfdndT)) then
      t25 = 1d0/t23
   end if

   if (present(dfdV) .or. present(dfdV2) .or. present(dfdTdV) .or. present(dfdndV)) then
      t26 = -2*V - 2*t12
   end if

   if (present(dfdV) .or. present(dfdn) .or. present(dfdTdV) .or. present(dfdndT)) then
      t27 = t22**(-2)
   end if

   if (present(dfdV2) .or. present(dfdndn) .or. present(dfdndV)) then
      t28 = t23**(-3)
   end if

   if (present(dfdV2) .or. present(dfdndn) .or. present(dfdndV)) then
      t29 = t22**(-3)
   end if

   if (present(f) .or. present(dfdT)) then
      t30 = 1d0/t22
   end if

   if (present(f)) then
      res = R*T*t30 - t13*t25
      f = res
   end if

   if (present(dfdT)) then
      res = R*t30 - t14*t25
      dfdT = res
   end if

   if (present(dfdV)) then
      res = -R*T*t27 - t13*t24*t26
      dfdV = res
   end if

   if (present(dfdn)) then
      do i = 1, nc
         res = R*T*t27*b(i) - t13*t24*(t12*b(i) - t22*b(i) + t7(i)) - t25*(t15(i &
               ) + t16(i))
         dfdn(i) = res
      end do
   end if

   if (present(dfdT2)) then
      res = -t17*t25
      dfdT2 = res
   end if

   if (present(dfdV2)) then
      res = 2*R*T*t29 + 2*t13*t24 - t13*t26*t28*(-4*V - 4*t12)
      dfdV2 = res
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
            res = 2*R*T*t29*b(i)*b(j) - 2*t13*t24*b(i)*b(j) - t13*t28*(t12*b(i) - &
                  t22*b(i) + t7(i))*(2*t12*b(j) - 2*t22*b(j) + t11(j)) - t24*(t15(i &
                  ) + t16(i))*(t12*b(j) - t22*b(j) + t7(j)) - t24*(t18(j) + t19(j)) &
                  *(t12*b(i) - t22*b(i) + t7(i)) - t25*(t3(i, j)*t4(i, j) + t3(i, j &
                  )*t6(i, j))
            dfdndn(i, j) = res
         end do
      end do
   end if

   if (present(dfdTdV)) then
      res = -R*t27 - t14*t24*t26
      dfdTdV = res
   end if

   if (present(dfdndT)) then
      do i = 1, nc
         res = R*t27*b(i) - t14*t24*(t12*b(i) - t22*b(i) + t7(i)) - t25*(t20(i) + &
               t21(i))
         dfdndT(i) = res
      end do
   end if

   if (present(dfdndV)) then
      do i = 1, nc
         res = -2*R*T*t29*b(i) + 2*t13*t24*b(i) - t13*t26*t28*(2*t12*b(i) - 2*t22 &
               *b(i) + t11(i)) - t24*t26*(t15(i) + t16(i))
         dfdndV(i) = res
      end do
   end if

end subroutine pr_e2_w6
