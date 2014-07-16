#!/usr/bin/perl
#
# perl build_cluster_hierarchy.pl <hidden_layer_trace> <vocabulary_size> <lvl1_size> [ <lvl2_size> [...] ]
#
# Gwénolé Lecorvé
# Idiap
# 2011-2012
#

use File::Temp qw/tempdir/;
use Cwd 'abs_path';

my $RNNLM_KMEANS_N_CORES = `nproc`;
chomp $RNNLM_KMEANS_N_CORES;
my $HERE = `dirname $0`;
chomp $HERE;
my $KMEANS = "$HERE/../../kmeans/omp_main -p $RNNLM_KMEANS_N_CORES";
my $LAPLACE_ADD = 5;
my $TEMP_DIR = "/tmp";

my $FULL_TRACE = abs_path(shift(@ARGV));
my $VOCAB_SIZE = shift(@ARGV);


my $tmpdir = tempdir(DIR => $TEMP_DIR, CLEANUP => 1);


sub log10 {
my $n = shift;
return log($n)/log(10);
}

sub compute_word_probs {
	my $hfn = shift;
	my $wfn = shift;
	my $p_prob = shift;
	my %freq = ();
	my %total = ();
	open(H, $hfn);
	open(W, $wfn);
	while (($hl = <H>) && ($wl = <W>)) {
		chomp $hl;
		chomp $wl;
		my @htab = split(/[ \t]/, $hl);
		my @wtab = split(/[ \t]/, $wl);
		$freq{$htab[1]}{$wtab[1]}++;
		$total{$htab[1]}++;
	}
	close(H);
	close(W);
	

	
	foreach my $k (keys(%freq)) {
		my $denom = $total{$k};
		for (my $w=0; $w < $VOCAB_SIZE; $w++) {
#			$$p_prob{$k}{$w} = $freq{$k}{$w} / $denom; # no smoothing
			$$p_prob{$k}{$w} = ($LAPLACE_ADD + $freq{$k}{$w}) / ($denom+$LAPLACE_ADD*$VOCAB_SIZE); # laplace smoothing
		}
	}
}

sub compute_cluster {
	my $s = shift;
	my $f = shift;
	
	system("$KMEANS -n $s -i $f > /dev/null");
	
	# cluster prior probs
	open(F, "$f.membership");
	my $total = 0;
	my @freq = ();
	while (<F>) {
		chomp;
		$_ =~ /^(\d+) (\d+)$/;
		$freq[$2]++;
		$total++;
	}
	close(F);
	
	# cluster-conditioned word probs
	my %w_probs = ();
	compute_word_probs("$f.membership",$WRD_TRACE, \%w_probs);
	
	# writing cluster definitions
	open(F, "$f.cluster_centres");
	my $k = 0;
	while (<F>) {
		chomp;
		$_ =~ /^(\d+) (.*)$/;
		# cluster prob and mean
		print sprintf("%.6f",$freq[$1]/$total)." $2\n";
		foreach my $w (sort {$a <=> $b} (keys(%{$w_probs{$k}}))) {
			print log($w_probs{$k}{$w})."\n";
		}
		# word probs
		
		$k++; # next cluster
	}
	

	
	print "--\n";
	close(F);
	
}


######### MAIN #########

foreach my $size (@ARGV) {
	system("ln -s $FULL_TRACE $tmpdir/trace.$size");
	print STDERR "Computing $size means...\n";
	compute_cluster($size, "$tmpdir/trace.$size");
}




