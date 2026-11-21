subroutine molefrac_p22_g0(nc, n, f)
   implicit none
   integer, intent(in) :: nc
   real(8), intent(in) :: n(:)
   real(8), optional, intent(out) :: f(:)
   real(8) :: res, t1, t2
   integer :: k, l

   if (present(f)) then
      t1 = 0.0d0
      do k = 1, nc
         t1 = t1 + n(k)
      end do
   end if

   if (present(f)) then
      t2 = 1d0/t1
   end if

   if (present(f)) then
      do l = 1, nc
         res = t2*n(l)
         f(l) = res
      end do
   end if

end subroutine molefrac_p22_g0
