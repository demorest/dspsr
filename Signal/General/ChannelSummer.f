C ChannelSummer- fortran version of the ChannelSum algorithm
C Original code by HSK 4 April 2004
C ******************************************************************
      SUBROUTINE channelsummer(chans_per_group, todo,
     :                         input, output, offsets)
C ******************************************************************
      implicit none

      integer chans_per_group, todo
      real input(*), output(*)
      integer offsets(chans_per_group)
      integer BB, nblocks, leftovers, I, J, K
      integer jump, jump2

c Algorithm 3 with no bad channels:
      BB = 8192

      nblocks = todo/BB
      leftovers = todo - nblocks*BB

c      write (*,*) "Got ", nblocks," ",leftovers

      DO I=1,nblocks
         jump = (I-1)*BB
         DO K=1,BB
	    output(K+jump) = 0.0
         ENDDO
         
         DO J=1,chans_per_group
            jump2 = offsets(J) + jump
            DO K=1,BB
               output(K+jump) = output(K+jump) + 
     +              input(jump2 + K)
            ENDDO
         ENDDO

      ENDDO
        
      jump = nblocks*BB

      DO K=1,leftovers
        output(K+jump) = 0.0
      ENDDO      

      DO J=1,chans_per_group
         jump2 = offsets(J) + jump
         DO K=1,leftovers
            output(K+jump) = output(K+jump) + 
     +           input(jump2 + K)
         ENDDO
      ENDDO

      RETURN
C
C END OF SUBROUTINE channelsummer
C
      END
