
# import modules from other libraries
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import logging
import tablib
import argparse 

CPI_DATA_URL = 'http://research.stlouisfed.org/fred2/data/CPIAUCSL.txt'

class CPIData(object): #Class

	"""Abstraction of the CPI data provided by FRED.
    This stores internally only one value per year.
    """

	def __init__(self): #Methods
		self.year_cpi = {}
		self.last_year = None
		self.first_year = None

	def load_from_url (self, url, save_as_file = None):
		"""Loads data from a given url. After fetching the file this implementation uses load_from_file
        internally.
        """

		fp = requests.get(url, stream = True, headers = {'Accept-Encoding': None}).raw

		if save_as_file is None:
			return self.load_from_file(fp)

		else:
			with open(save_as_file, 'wb+') as out:
				while True:
					buffer = fp.read(81920)
					if not buffer:
						break
					out.write(buffer)
			with open(save_as_file) as fp:
				return self.load_from_file(fp)

	def load_from_file (self, fp):
		"""Loads CPI data from a given file-like object.
		"""

		current_year = None
		year_cpi = []
		for line in fp:
			while not line.startswith("DATE "):
				pass

			data = line.rstrip().split()

			year = int(data[0].split("-")[0])
			cpi = float(data[1])

			if self.first_year is None:
				self.first_year = year

			self.last_year = year

			if current_year != year:
				if current_year is not None:
					self.year_cpi[current_year] = sum(year_cpi) / len(year_cpi)
				year_cpi = []
				current_year = year
			
			year_cpi.append(cpi)

		if current_year is not None and current_year not in self.year_cpi:
			self.year_cpi[current_year] = sum(year_cpi) / len(year_cpi)

	def get_adjusted_price (self, price, year, current_year = None):
		"""Returns the adapted price from a given year compared to what current
        year has been specified.
        This essentially is the calculated inflation for an item.
        """

		if current_year is None or current_year > 2013:
			current_year = 2013

		if year < self.first_year:
		    year = self.first_year
		elif year > self.last_year:
			year = self.last_year

		year_cpi = self.year_cpi[year]
		current_cpi = self.year_cpi[current_year]
		
		return float(price) / year_cpi * current_cpi

class GiantbombAPI(object):
	"""Generator yielding platforms matching the given criteria. If no
        limit is specified, this will return *all* platforms.
    """
	
	base_url = 'http://www.giantbomb.com/api/'

	def __init__(self, api_key):
		self.api_key = api_key

	def get_Plaforms(self, sort = None, filter = None, field_list = None):

		# The following lines also do value-format conversions from what's
        # common in Python (lists, dictionaries) into what the API requires.
		params = {}
		if sort is not None:
			params['sort'] = sort
		if field_list is not None:
			params['field_list'] = ','.join(field_list)
		if filter is not None:
			params['filter'] = filter
			for key,value in filter.iteritems():
				parsed_filters.append('{0}:{1}'.format(key, value))
			params['filter'] = ','.join(parsed_filters)

		# append API key to the list of parameters
		params['api_key'] = self.api_key
		params['format'] = 'json' 

		# Giantbomb's limit for items in a result set for this API is 100
        # items. But given that there are more than 100 platforms in their
        # database we will have to fetch them in more than one call.
		incomplete_result = True
		num_total_results = None
		num_fetched_results = 0
		Counter = 0

		while incomplete_result:
			params['offset'] = num_fetched_results
			result = requests.get(self.base_url + '/ platforms', params = params)
			result = result.json()

			if num_total_results is None:
				num_total_results = int(result['number_of_total_results'])
			num_fetched_results += int(result['number_of_page_results'])
			if num_fetched_results >= num_total_results:
				incomplete_result = False
			for item in result['results']:
				logging.debug("Yielding platform {0} of {1}".format(counter + 1, num_total_results))	

			if 'original_price' in item and item['original_price']:
				item['original_price'] = float(item['original_price'])

			yield item
			Counter += 1 


def is_valid_dataset(platform): #Function

	"""Filters out datasets that we can't use since they are either lacking
    a release date or an original price. For rendering the output we also
    require the name and abbreviation of the platform.

    """

	if 'release_date' not in platform or not platform['release_date']:
		logging.warn(u"{0} has not release_date".format(platform['name']))
		return False
	if 'original_price' not in platform or not platform['original_price']:
		logging.warn(u"{0} has no original_price".format(platform['name']))
		return False
	if 'name' not in platform or not platform['name']:
		logging.warn(u"{0} has no platform name found for given dataset")
		return False
	if 'abbreviation' not in platform or not platform['abbreviation']:
		logging.warn(u"{0} has no abbreviation".format(platform['name']))
		return False
	return True


def generate_plot(platforms, output_file): #Function

	"""Generates a bar chart out of the given platforms and writes the output
    into the specified file as PNG image.
    """

	labels = []
	values = []

	for platform in platforms:
		name = platform['name']
		adapted_price = platform['adjusted_price']
		price = platform['original_price']

		if price > 2000:
			continue

		if len(name) > 15:
			name = platform['abbreviation']
		labels.insert(0, u"{0}\n$ {1}\n$ {2}".format(
			name, price, round(adjusted_price, 2)))
		values.insert(0, adapted_price)

	width = 0.3
	ind = np.arange(len(values))
	fig = plt.figure(figsize = (len(labels) * 1.8, 10))

	ax = fig.add_subplot(1, 1, 1)
	ax.bar(ind, values, width, align = 'center')

	plt.ylabel('adjusted_price')
	plt.xlabel('Year / Console')
	ax.set_xticks(ind + 0.3)
	ax.set_xticklabels(labels)
	fig.autofmt_xdate()
	plt.grid(True)

	plt.show(dpi = 72)

def generate_csv(platforms, output_file): #Function
	
	"""Writes the given platforms into a CSV file specified by the output_file
    parameter.
    """
	dataset = tablib.Dataset(headers = ['Abbreviation', 'Name', 'Year', 'Price', 'Adjusted Price'])

	for p in platforms:
		dataset.append([p['abbreviation'], p['name'], p['year'], p['original_price'], p['adjusted_price']])

	# If the output_file is a string it represents a path to a file which
    # we will have to open first for writing. Otherwise we just assume that
    # it is already a file-like object and write the data into it.
    	
	if isinstance(output_file, basestring):
		with open(output_file, 'w+') as fp:
			fp.write(dataset.csv)
	else:
		output_file.write(dataset.csv)

def parse_args(): #Function
	
	"""awesome module for adding parameters to the classes in your script
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('--giantbomb-api-key', required = True, 
						help = 'API Key provided by Giantbomb.com')
	parser.add_argument('--cpi-file',
						default = os.path.join(os.path.dirname(__file__),
											'CPIAUSCL.txt'),
											help = 'Path to file containt the CPI data')
	parser.add_argument('--cpi-data-url', default = CPI_DATA_URL,
						help = 'URL which should be used as CPI data source')
	parser.add_argument('--debug', default = False, action = 'store_true',
						help = 'debug')
	parser.add_argument('--csv-file', help = 'Path to CSV file for output')
	parser.add_argument('--plot-file', help = 'Path to PNG file for output')
	parser.add_argument('--limit', type = int, help = "Number of recent platforms to be considered")
	opts = parser.parse_args()

	if not (opts.plot_file or opts.csv_file):
			parser.error("You have to specify either a --csv-file or --plot-file")
	return opts

def main():

	"""this function handles the logic of the script"""

	opts = parse_args()

	if opts.debug:
		logging.basicConfig(level = logging.DEBUG)
	else:
		logging.basicConfig(level = logging.INFO)

	cpi_data = CPIData()
	gb_api = GiantbombAPI(opts.giantbomb_api_key)

	if os.path.exists(opts.cpi_file):
		with open(opts.cpi_file) as fp:
			cpi_data.load_from_file(fp)
	else:
		cpi_data.load_from_url(opts.cpi_data_url, save_as_file = opts.cpi_file)

	platforms = []
	counter = 0

	for platform in gb_api.get_Plaforms(sort = 'release_date:decs',
										field_list = ['release_date',
														'original_price', 'name',
														'abbreviation']):
		if not is_valid_dataset(platform):
			continue

		year = int(platform['release_date'].split('-')[0])
		price = platform['original_price']
		adjusted_price = cpi_data.get_adjusted_price(price,year)
		platform['year'] = year
		platform['original_price'] = price
		platform['adjusted_price'] = adjusted_price
		platforms.append(platform)

		if opts.limit is not None and counter + 1 >= opts.limit:
			break
		counter += 1

	if opts.plot_file:
		generate_plot(platform, opts.plot_file)
	if opts.csv_file:
		generate_csv(platform, opts.csv_file)

if __name__ == '__main__':
	main()