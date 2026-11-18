subroutine molefrac_e8_g3(nc, n, f, dfdn, dfdndn)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f(:)
   real(8), optional, intent(out) :: dfdn(:, :)
   real(8), optional, intent(out) :: dfdndn(:, :, :)
   real(8) :: res, t1, t2, t3, t4, t5
   integer :: i, j, k, l

   if (present(f) .or. present(dfdn) .or. present(dfdndn)) then
      t1 = 0.0d0
      do k = 1, nc
         t1 = t1 + n(k)
      end do
   end if

   if (present(dfdn) .or. present(dfdndn)) then
      t2 = t1**(-2)
   end if

   if (present(dfdndn)) then
      t3 = t1**(-3)
   end if

   if (present(dfdndn)) then
      t4 = -t2
   end if

   if (present(f) .or. present(dfdn)) then
      t5 = 1d0/t1
   end if

   if (present(f)) then
      do l = 1, nc
         res = t5*n(l)
         f(l) = res
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         do l = 1, nc
            if (i == l) then
               res = -t2*n(i) + t5
            else
               res = -t2*n(l)
            end if
            dfdn(i, l) = res
         end do
      end do
   end if

   if (present(dfdndn)) then
      do i = 1, nc
         do j = 1, nc
            do l = 1, nc
               if (i == j .and. i == l) then
                  res = -2*t2 + 2*t3*n(i)
               else if (i == l) then
                  res = 2*t3*n(i) + t4
               else if (j == l) then
                  res = 2*t3*n(j) + t4
               else
                  res = 2*t3*n(l)
               end if
               dfdndn(i, j, l) = res
            end do
         end do
      end do
   end if

end subroutine molefrac_e8_g3
