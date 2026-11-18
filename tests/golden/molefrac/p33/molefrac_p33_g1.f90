subroutine molefrac_p33_g1(nc, n, f, dfdn)
   implicit none
   integer, intent(in) :: nc
   real(8), contiguous, intent(in) :: n(:)
   real(8), contiguous, optional, intent(out) :: f(:)
   real(8), contiguous, optional, intent(out) :: dfdn(:, :)
   real(8) :: res, t1, t2, t3
   integer :: i, k, l

   if (present(f) .or. present(dfdn)) then
      t1 = 0.0d0
      do k = 1, nc
         t1 = t1 + n(k)
      end do
   end if

   if (present(dfdn)) then
      t2 = t1**(-2)
   end if

   if (present(f) .or. present(dfdn)) then
      t3 = 1d0/t1
   end if

   if (present(f)) then
      do l = 1, nc
         res = t3*n(l)
         f(l) = res
      end do
   end if

   if (present(dfdn)) then
      do i = 1, nc
         do l = 1, nc
            if (i == l) then
               res = -t2*n(i) + t3
            else
               res = -t2*n(l)
            end if
            dfdn(i, l) = res
         end do
      end do
   end if

end subroutine molefrac_p33_g1
